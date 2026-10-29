"""Train the Trex walking policy.

    conda activate isaac6; export OMNI_KIT_ACCEPT_EULA=YES
    python walk_rl/train.py --name run0 --iterations 1000

Outputs (outputs/walk_runs/<name>/): train_log.jsonl, checkpoint.pt (resume with
--resume), policy_numpy.npz (deployable actor: normalizer + MLP, plain NumPy).
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="run0")
parser.add_argument("--iterations", type=int, default=1000)
parser.add_argument("--num-envs", type=int, default=None, help="override walk_task.yaml ppo.num_envs")
parser.add_argument("--task", default=str(PROJECT_ROOT / "walk_rl" / "walk_task.yaml"))
parser.add_argument("--resume", action="store_true")
parser.add_argument("--reset-lr", action="store_true", help="on --resume, restart at ppo.lr instead of the saved LR")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

import yaml  # noqa: E402

from walk_rl.env import TrexWalkEnv, load_task_config, ns_to_dict  # noqa: E402
from walk_rl.ppo import PPO  # noqa: E402

try:
    cfg = load_task_config(args.task)
    num_envs = args.num_envs or cfg.ppo.num_envs
    log_dir = PROJECT_ROOT / "outputs" / "walk_runs" / args.name
    log_dir.mkdir(parents=True, exist_ok=True)
    # reproducibility: freeze the MERGED config (defaults + --task overrides) actually used
    with open(log_dir / "walk_task.yaml", "w") as f:
        yaml.safe_dump(ns_to_dict(cfg), f, sort_keys=False)
    env = TrexWalkEnv(num_envs, cfg, simulation_app=simulation_app, seed=args.seed)
    print(f"[train] task={args.task}", flush=True)
    print(f"[train] envs={num_envs} actor_obs={env.num_actor_obs} critic_obs={env.num_critic_obs} "
          f"control={cfg.sim.control_hz}Hz DR={env.dr_active}", flush=True)
    ppo = PPO(env, cfg.ppo, log_dir, device=env.device)
    if args.resume and (log_dir / "checkpoint.pt").exists():
        ppo.load(log_dir / "checkpoint.pt", reset_lr=args.reset_lr)
        print(f"[train] resumed from iteration {ppo.iteration}", flush=True)
    meta = {"control_hz": cfg.sim.control_hz, "actuated_joints": env.act_names, "actor_obs_dim": env.num_actor_obs,
            "joint_limit_rad": env.joint_limit.tolist(), "act_scale_rad": env.act_scale.tolist(),
            "servo_resolution_rad": env.servo_res, "use_phase_clock": env.use_clock,
            "gait_frequency_hz": cfg.gait.frequency_hz,
            "gait_frequency_per_mps": cfg.gait.frequency_per_mps, "gait_frequency_ref_mps": cfg.gait.frequency_ref_mps,
            "gait_frequency_range_hz": cfg.gait.frequency_range_hz, "smoothing_alpha": cfg.actions.smoothing_alpha,
            "obs_layout": env.obs_layout()}
    ppo.learn(args.iterations, extra_meta=meta)
except BaseException:
    # Kit's fast shutdown in close() exits before Python prints a pending traceback.
    import traceback

    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    simulation_app.close()
