"""Pi-runner pass tests 1 and 2: the runner reproduces the training observations and servo
goals from raw sensor readings, and its NumPy policy matches the torch actor.

Runs the Isaac Sim env on CPU physics with sensor noise, encoder offsets, delays and DR off
(those are what the hardware adds; the runner must not add or remove anything). Each step
the env's true sensor values (gravity, gyro, yaw, joint positions) go through the runner;
its observation, action and servo goal are compared with the env's.

    python pi_runner/test_obs_parity.py --run v8d          # conda env isaac6
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--run", default="v8d")
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--cmd", type=float, nargs="+", default=[0.0, 0.10, 0.20])
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import torch  # noqa: E402

try:
    from pi_runner.runner import PolicyRunner, Reading
    from walk_rl.env import TrexWalkEnv
    from walk_rl.ppo import ActorCritic
    from walk_rl.task_core import apply_checkpoint_interface, load_task_config

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
    ob.imu_mount_offset_deg = 0.0
    for k in vars(ob.noise):
        setattr(ob.noise, k, 0.0)
    if ob.heading.enabled:
        ob.heading.drift_deg_per_min = 0.0
        ob.heading.noise_rad = 0.0  # ref_init_range stays: the runner gets the latched reference

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
    runners = []
    for i in range(n):
        r = PolicyRunner(run_dir)
        r.reset_state(env.cmd_target[i].numpy(), phase=float(env.phase[i]), yaw_ref=float(env.yaw_ref[i]))
        runners.append(r)

    def reading(i):
        return Reading(gravity_b=env.gravity_b[i].numpy().astype(np.float64), gyro_b=env.ang_vel_b[i].numpy().astype(np.float64),
                       yaw=float(env._yaw()[i]), q=env.q[i, env.act_idx].numpy().astype(np.float64))

    err = {"obs": 0.0, "action_np_vs_torch": 0.0, "goal_ticks": 0.0}
    tick_diffs, tick_total = 0, 0
    live = np.ones(n, dtype=bool)
    steps_cmp = np.zeros(n, dtype=int)
    for t in range(args.steps):
        with torch.no_grad():
            act = ac.act_deterministic(obs["actor"])
        goals = []
        for i in range(n):
            if not live[i]:
                goals.append(None)
                continue
            g = runners[i].step(reading(i), float(cmds[i]))
            goals.append(g)
            err["obs"] = max(err["obs"], float(np.abs(runners[i].last_obs - obs["actor"][i].numpy()).max()))
            # NumPy policy vs torch actor on the SAME input (the env's float32 observation)
            a_np = runners[i].policy(obs["actor"][i].numpy().astype(np.float64))
            err["action_np_vs_torch"] = max(err["action_np_vs_torch"], float(np.abs(a_np - act[i].numpy()).max()))
        obs, rew, done, _ = env.step(act)
        for i in range(n):
            if live[i]:
                # in servo ticks (0.24 deg); the runner computes in float32 like training, so goals must be identical
                ticks = np.abs(goals[i] - env.cmd_target[i].numpy()) / runners[i].servo_res
                err["goal_ticks"] = max(err["goal_ticks"], float(ticks.max()))
                tick_diffs += int((ticks > 0.25).sum())
                tick_total += ticks.size
                steps_cmp[i] += 1
        live &= ~done.numpy()
        if not live.any():
            break
    print(f"[parity] run={args.run} cmds={args.cmd} steps compared per env={steps_cmp.tolist()}")
    for k, v in err.items():
        print(f"[parity] max |runner - env| {k:20s} {v:.2e}")
    print(f"[parity] servo goals that differ: {tick_diffs}/{tick_total}")
    # The Pi runs the NumPy policy, training the torch one: their ~1e-6 action difference can land a goal
    # on the other side of a 0.24 deg servo step. That is allowed as 1 tick in at most 1 of 1000 goals
    # (v9k: 1/4800); anything more is a real mismatch.
    ok = (err["obs"] < 1e-4 and err["action_np_vs_torch"] < 1e-5 and err["goal_ticks"] <= 1.0 + 1e-6
          and tick_diffs <= tick_total // 1000 and steps_cmp.min() >= 50)
    print(f"[parity] {'OK' if ok else 'FAILED'} (pass: obs < 1e-4, NumPy vs torch < 1e-5 on the same input, "
          f"servo goals identical except 1-tick rounding in <= 0.1 %, >= 50 steps)")
except Exception:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
finally:
    app.close()
