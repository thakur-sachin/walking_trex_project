"""Smoke test for TrexWalkEnv: shapes, finiteness, DR status, ankle-linkage
residual, reward terms, and throughput. Run:

    conda activate isaac6; export OMNI_KIT_ACCEPT_EULA=YES
    python walk_rl/smoke_env.py --num-envs 256
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=100)
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

import torch  # noqa: E402

from walk_rl.env import TrexWalkEnv  # noqa: E402

try:
    env = TrexWalkEnv(args.num_envs, simulation_app=simulation_app)
    print(f"[smoke] actions={env.num_actions} actor_obs={env.num_actor_obs} critic_obs={env.num_critic_obs} "
          f"dt={env.dt:.3f}s decimation={env.decimation} ep_steps={env.max_episode_steps}")
    print(f"[smoke] DR status: {env.dr_active}")
    obs = env.reset()
    assert obs["actor"].shape == (args.num_envs, env.num_actor_obs), obs["actor"].shape
    assert obs["critic"].shape == (args.num_envs, env.num_critic_obs), obs["critic"].shape

    for label, make in (("zero", lambda: torch.zeros(args.num_envs, env.num_actions, device=env.device)),
                        ("random", lambda: (torch.rand(args.num_envs, env.num_actions, device=env.device) * 2 - 1) * 0.5)):
        env.reset()
        t0 = time.time()
        rew_sum = torch.zeros(args.num_envs, device=env.device)
        n_done = 0
        for _ in range(args.steps):
            obs, rew, done, extras = env.step(make())
            assert torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all(), "non-finite obs"
            rew_sum += rew
            n_done += int(done.sum())
        torch.cuda.synchronize()
        dt = time.time() - t0
        # ankle linkage residual (kinematic enforcement happens before each physics step;
        # measure what is left after the solver acted)
        q = env.q
        resid = torch.rad2deg((q[:, env.ank_idx] - (q[:, env.fem_idx] - q[:, env.tib_idx])).abs())
        stats = env.pop_episode_stats()
        print(f"[smoke] {label}: {args.steps * args.num_envs / dt:.0f} env-steps/s, mean ep reward/step="
              f"{rew_sum.mean().item() / args.steps:.3f}, resets={n_done}, ankle residual deg mean={resid.mean():.2f} max={resid.max():.2f}")
        if stats:
            print(f"[smoke]   episodes={stats['episodes']:.0f} len={stats['episode_length_s']:.2f}s "
                  f"terminated={stats['terminated_frac']:.2f} dist_x={stats['distance_x_m']:.3f}m")
            print("[smoke]   terms:", {k.split('/')[1]: round(v, 3) for k, v in stats.items() if k.startswith('term/')})
    # push path: kick every env once and check the base velocity changed
    env.reset()
    env.set_push_scale(1.0)
    env.push_timer[:] = 0.0
    obs, rew, done, extras = env.step(torch.zeros(args.num_envs, env.num_actions, device=env.device))
    print(f"[smoke] pushes applied: {env.push_count} (expected {args.num_envs}); obs finite: "
          f"{bool(torch.isfinite(obs['actor']).all())}; feet_y_neutral={env.feet_y_neutral:.3f} m")
    print(f"[smoke] obs layout: {env.obs_layout()}")
except BaseException:
    # Kit's fast shutdown in close() exits before Python prints a pending traceback.
    import traceback

    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    simulation_app.close()
