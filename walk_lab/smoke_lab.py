"""Smoke test for the Isaac Lab port: builds TrexLabWalkEnv, dumps its physics settings
(diff against walk_lab/physics_dump.py output from the Isaac Sim stack), and runs a trained
policy with the same schedule as walk_rl/trace_env.py.

    conda activate isaaclab6; export OMNI_KIT_ACCEPT_EULA=YES
    python walk_lab/smoke_lab.py --device cpu --num-envs 16 --dump /tmp/phys_lab.json --out /tmp/trace_lab.npz
"""
import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--run", default="v8d")
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--device", default="cpu")
parser.add_argument("--dump", default=None)
parser.add_argument("--harness", type=float, default=0.5, help="harness alpha for the first 40 steps")
parser.add_argument("--out", default=None)
parser.add_argument("--compare", default=None)
parser.add_argument("--actuator", choices=["implicit_pd", "lx16a"], default=None, help="override actuator.model")
args = parser.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True, device=args.device).app

import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

try:
    from walk_lab.env import TrexLabWalkEnv
    from walk_rl.ppo import ActorCritic
    from walk_rl.task_core import apply_checkpoint_interface, load_task_config

    run_dir = PROJECT_ROOT / "outputs" / "walk_runs" / args.run
    cfg = load_task_config(run_dir / "walk_task.yaml")
    ck = torch.load(run_dir / "checkpoint.pt", map_location="cpu")
    apply_checkpoint_interface(cfg, ck["meta"])
    if args.actuator:
        cfg.actuator.model = args.actuator
    cfg.push.interval_s = [0.4, 0.8]
    cfg.sim.episode_length_s = 2.0
    t0 = time.time()
    env = TrexLabWalkEnv(args.num_envs, cfg, device=args.device, seed=7)
    print(f"[lab] built in {time.time() - t0:.1f}s: actions={env.num_actions} actor_obs={env.num_actor_obs} "
          f"critic_obs={env.num_critic_obs} dt={env.dt} decimation={env.decimation} DR={env.dr_active}")
    print(f"[lab] dofs={env.dof_names}\n[lab] links={env.link_names}")
    if args.dump:
        import omni.usd

        from walk_lab.physics_dump import dump_stage_physics
        d = dump_stage_physics(omni.usd.get_context().get_stage(), "/World/envs/env_0/Robot")
        json.dump(d, open(args.dump, "w"), indent=1, sort_keys=True)
        print(f"[lab] physics dump -> {args.dump}")

    ac = ActorCritic(env.num_actor_obs, env.num_critic_obs, env.num_actions, cfg.ppo.actor_hidden,
                     cfg.ppo.critic_hidden, cfg.ppo.init_log_std).to(env.device)
    ac.load_state_dict(ck["model"])
    torch.manual_seed(11)
    obs = env.reset()
    assert obs["actor"].shape == (args.num_envs, env.num_actor_obs) and obs["critic"].shape == (args.num_envs, env.num_critic_obs)
    rec = {k: [] for k in ("actor", "critic", "reward", "done", "root_pos", "q")}
    t0 = time.time()
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
        assert torch.isfinite(obs["actor"]).all() and torch.isfinite(rew).all()
        rec["actor"].append(obs["actor"].double().sum().item())
        rec["critic"].append(obs["critic"].double().sum().item())
        rec["reward"].append(rew.double().sum().item())
        rec["done"].append(done.sum().item())
        rec["root_pos"].append(env.root_pos.double().sum().item())
        rec["q"].append(env.q.double().sum().item())
    el = time.time() - t0
    out = {k: np.asarray(v) for k, v in rec.items()}
    print(f"[lab] {args.steps} steps in {el:.1f}s, dones={int(out['done'].sum())}, pushes={env.push_count}, "
          f"reward_sum={out['reward'].sum():.3f}, episodes={env.pop_episode_stats()}")
    if args.out:
        np.savez(args.out, **out)
    if args.compare:
        ref = np.load(args.compare)
        for k in out:
            print(f"[lab] vs isaacsim {k:9s} first10 lab={np.round(out[k][:5], 3)} sim={np.round(ref[k][:5], 3)}  "
                  f"mean |diff|/|ref| = {np.abs(out[k] - ref[k]).mean() / (np.abs(ref[k]).mean() + 1e-9):.3f}")
    env.close()
except Exception:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
finally:
    app.close()
