"""Ankle linkage model check: teleport (v1-v8) vs teleport_routed (robot_config
ankle_linkage sim_model), on CPU physics so it can run beside a GPU training job.

Runs a trained policy (deterministic, DR / pushes / delays / noise off) at a few commands
and reports, per model:
  - survival and forward speed
  - linkage error |ankle - (femur - tibia)| at the end of each control step (mean, max)
  - servo load on femur and tibia, mean |torque| / stall: the joint force PhysX reports,
    and the same minus the routed linkage effort (the servo's own share)
  - the routed ankle torque itself, mean |tau_a| / stall

    python walk_rl/test_ankle_linkage.py --run v8d --model teleport_routed [--inertia scale_with_mass]
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--run", default="v8d")
parser.add_argument("--model", choices=["teleport", "teleport_routed"], required=True)
parser.add_argument("--inertia", choices=["urdf", "scale_with_mass"], default=None, help="override robot_config")
parser.add_argument("--cmd", type=float, nargs="+", default=[0.0, 0.10, 0.20])
parser.add_argument("--physics-hz", type=int, default=None, help="override the task's physics rate")
parser.add_argument("--debug", type=int, default=0, help="print state for the first N steps")
parser.add_argument("--seconds", type=float, default=10.0)
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import torch  # noqa: E402

from common.robot_config import RobotConfig  # noqa: E402

_load = RobotConfig.load.__func__


def _patched(cls, path):
    rc = _load(cls, path)
    rc.ankle_sim_model = args.model
    if args.inertia:
        rc.base_link_inertia = args.inertia
    return rc


RobotConfig.load = classmethod(_patched)

try:
    from walk_rl.env import TrexWalkEnv, apply_checkpoint_interface, load_task_config
    from walk_rl.ppo import ActorCritic

    run_dir = PROJECT_ROOT / "outputs" / "walk_runs" / args.run
    cfg = load_task_config(run_dir / "walk_task.yaml")
    ck = torch.load(run_dir / "checkpoint.pt", map_location="cpu")
    apply_checkpoint_interface(cfg, ck["meta"])
    cfg.domain_randomization.enabled = False
    cfg.push.enabled = False
    cfg.harness.enabled = False
    cfg.actions.delay_steps_range = [0, 0]
    ob = cfg.observation
    ob.imu_delay_steps_range = [0, 0]
    ob.encoder_offset_rad = 0.0
    for k in vars(ob.noise):
        setattr(ob.noise, k, 0.0)
    cfg.sim.episode_length_s = args.seconds + 1.0
    if args.physics_hz:
        cfg.sim.physics_hz = args.physics_hz

    n = len(args.cmd)
    env = TrexWalkEnv(n, cfg, device="cpu", simulation_app=app, seed=3)
    ac = ActorCritic(env.num_actor_obs, env.num_critic_obs, env.num_actions, cfg.ppo.actor_hidden,
                     cfg.ppo.critic_hidden, cfg.ppo.init_log_std)
    ac.load_state_dict(ck["model"])
    cmds = torch.tensor(args.cmd)

    def fixed(ids):
        env.cmd_vx[ids] = cmds[ids]
        env.cmd_timer[ids] = 1e9
    env._sample_commands = fixed
    obs = env.reset()

    fem_tib = [env.act_names.index(f"{s}_{p}_joint") for p in ("femur", "tibia") for s in ("left", "right")]
    steps = int(args.seconds * cfg.sim.control_hz)
    alive = torch.ones(n, dtype=torch.bool)
    t_alive = torch.zeros(n)
    err_sum, err_max = torch.zeros(n), torch.zeros(n)
    rep_sum, drv_sum, vx_sum, ank_sum = torch.zeros(n), torch.zeros(n), torch.zeros(n), torch.zeros(n)
    for t in range(steps):
        with torch.no_grad():
            act = ac.act_deterministic(obs["actor"])
        obs, rew, done, _ = env.step(act)
        if args.debug and t < args.debug:
            print(f"[debug] t={t} done={done.tolist()} torque={env.torque[0].tolist()} q={env.q[0].tolist()} "
                  f"base_z={env.root_pos[0, 2].item():.3f} raw_force={env._sim_joint_forces()[0].tolist()}")
        q, qd = env._sim_dof_state()
        e = (q[:, env.ank_idx] - (q[:, env.fem_idx] - q[:, env.tib_idx])).abs().max(dim=1).values
        own = env.torque  # the servo's share (task_core subtracts the routed effort)
        rep = env._sim_joint_forces()[:, env.act_idx]
        tau_a = env._sim_joint_forces()[:, env.ank_idx].abs().mean(dim=1)
        a = alive.float()
        err_sum += a * e
        err_max = torch.maximum(err_max, a * e)
        rep_sum += a * (rep[:, fem_tib].abs() / env.tau_max).mean(dim=1)
        drv_sum += a * (own[:, fem_tib].abs() / env.tau_max).mean(dim=1)
        ank_sum += a * tau_a / env.tau_max
        vx_sum += a * env.lin_vel_b[:, 0]
        t_alive += a
        alive &= ~done
    dt = 1.0 / cfg.sim.control_hz
    rc = env.robot_cfg
    print(f"[linkage] run={args.run} model={args.model} inertia={rc.base_link_inertia}")
    print("[linkage]  cmd | survived | vx     | err mean / max [deg] | femur+tibia |tau|/stall: reported / own | ankle |tau|/stall")
    for i in range(n):
        s = max(float(t_alive[i]), 1.0)
        print(f"[linkage] {float(cmds[i]):.2f} | {float(t_alive[i]) * dt:5.1f}s{'' if alive[i] else ' FELL'} | "
              f"{float(vx_sum[i]) / s:+.3f} | {float(err_sum[i]) / s * 57.2958:6.3f} / {float(err_max[i]) * 57.2958:6.3f} | "
              f"{float(rep_sum[i]) / s:.3f} / {float(drv_sum[i]) / s:.3f} | {float(ank_sum[i]) / s:.3f}")
except Exception:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
finally:
    app.close()
