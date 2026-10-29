"""Deterministic step trace of the walking env, for checking that a refactor (or a port to
another backend) reproduces the task exactly.

Runs a trained policy on CPU physics (deterministic) with every training-time mechanism
active: harness, frequent pushes, a manual kick, resets, delays and DR. Records per-step
sums of observations, rewards, dones and base position.

    python walk_rl/trace_env.py --out /tmp/trace_a.npz                 # record
    python walk_rl/trace_env.py --out /tmp/trace_b.npz --compare /tmp/trace_a.npz
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--run", default="v8d", help="outputs/walk_runs/<run> whose checkpoint.pt drives the env")
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--device", default="cpu")
parser.add_argument("--harness", type=float, default=0.5, help="harness alpha for the first 40 steps")
parser.add_argument("--out", required=True)
parser.add_argument("--compare", default=None, help="trace .npz to compare against")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import torch  # noqa: E402

from walk_rl.env import TrexWalkEnv, apply_checkpoint_interface, load_task_config  # noqa: E402
from walk_rl.ppo import ActorCritic  # noqa: E402

try:
    run_dir = PROJECT_ROOT / "outputs" / "walk_runs" / args.run
    cfg = load_task_config(run_dir / "walk_task.yaml")
    ck = torch.load(run_dir / "checkpoint.pt", map_location="cpu")
    apply_checkpoint_interface(cfg, ck["meta"])
    cfg.push.interval_s = [0.4, 0.8]  # many pushes inside a short trace
    cfg.sim.episode_length_s = 2.0  # several timeouts/resets inside the trace

    env = TrexWalkEnv(args.num_envs, cfg, device=args.device, simulation_app=simulation_app, seed=7)
    ac = ActorCritic(env.num_actor_obs, env.num_critic_obs, env.num_actions, cfg.ppo.actor_hidden,
                     cfg.ppo.critic_hidden, cfg.ppo.init_log_std).to(env.device)
    ac.load_state_dict(ck["model"])
    torch.manual_seed(11)
    obs = env.reset()
    rec = {k: [] for k in ("actor", "critic", "reward", "done", "root_pos", "q")}
    for t in range(args.steps):
        env.set_harness(args.harness if t < 40 else 0.0)
        env.set_push_scale(1.0 if t >= 20 else 0.0)
        if t == 60:
            ids = torch.arange(0, env.num_envs, 2, device=env.device)
            env.kick(ids, torch.tensor([0.2, -0.1, 0.0], device=env.device).expand(len(ids), 3),
                     torch.zeros(len(ids), 3, device=env.device))
        with torch.no_grad():
            a = ac.act_deterministic(obs["actor"])
        obs, rew, done, _ = env.step(a)
        rec["actor"].append(obs["actor"].double().sum().item())
        rec["critic"].append(obs["critic"].double().sum().item())
        rec["reward"].append(rew.double().sum().item())
        rec["done"].append(done.sum().item())
        rec["root_pos"].append(env.root_pos.double().sum().item())
        rec["q"].append(env.q.double().sum().item())
    out = {k: np.asarray(v) for k, v in rec.items()}
    np.savez(args.out, **out)
    print(f"[trace] wrote {args.out}: {args.steps} steps, dones={int(out['done'].sum())}, "
          f"pushes={env.push_count}, reward_sum={out['reward'].sum():.6f}")
    if args.compare:
        ref = np.load(args.compare)
        worst = 0.0
        for k in out:
            d = np.abs(out[k] - ref[k])
            first = int(np.argmax(d > 1e-6)) if (d > 1e-6).any() else -1
            worst = max(worst, float(d.max()))
            print(f"[trace] {k:9s} max|diff|={d.max():.3e}  first step >1e-6: {first}")
        print(f"[trace] {'IDENTICAL' if worst <= 1e-6 else 'DIFFERENT'} (worst {worst:.3e})")
except Exception:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
finally:
    simulation_app.close()
