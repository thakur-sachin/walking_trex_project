"""Build the Newton env, step it with zero and random actions, print state, contacts and timing.

    ~/trex_stack/IsaacLab_dev/.venv/bin/python walk_lab3/smoke.py --task walk_rl/experiments/v9s_*.yaml --num-envs 64
"""
import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--num-envs", type=int, default=64)
ap.add_argument("--steps", type=int, default=100)
a = ap.parse_args()

import torch  # noqa: E402
from isaaclab.app import launch_simulation  # noqa: E402

from walk_lab3.env import TrexLab3WalkEnv, make_env_cfg  # noqa: E402
from walk_rl.task_core import load_task_config  # noqa: E402

cfg = load_task_config(a.task)
with launch_simulation(make_env_cfg(cfg, a.task, a.num_envs, "cuda:0"), {"device": "cuda:0"}):
    env = TrexLab3WalkEnv(a.num_envs, cfg, task_yaml=a.task)
    print("dofs", env.dof_names)
    print("links", env.link_names)
    print("dr_active", env.dr_active)
    obs = env.reset()
    print("actor obs", tuple(obs["actor"].shape), "critic", tuple(obs["critic"].shape))
    t0 = time.time()
    for k in range(a.steps):
        act = torch.zeros(env.num_envs, env.num_actions, device=env.device) if k < a.steps // 2 else \
            0.3 * torch.randn(env.num_envs, env.num_actions, device=env.device)
        obs, rew, done, info = env.step(act)
        if k % 25 == 0 or k == a.steps - 1:
            r = (env.q[:, env.ank_idx] - (env.q[:, env.fem_idx] - env.q[:, env.tib_idx])).abs().flatten() * 57.3
            ank = f"mean {r.mean():.2f} p99 {r.quantile(0.99):.2f} max {r.max():.2f}"
            print(f"step {k:3d} rew {rew.mean():+.3f} done {int(done.sum())} base_z {env.root_pos[:, 2].mean():.3f} "
                  f"foot_F {env.foot_force.mean(0).tolist()} ankle_res deg {ank}")
    dt = time.time() - t0
    print(f"{a.steps} steps x {env.num_envs} envs in {dt:.1f} s = {a.steps * env.num_envs / dt:.0f} env-steps/s")
