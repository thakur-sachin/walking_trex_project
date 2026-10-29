"""Self-collision check for a robot asset in either Isaac stack (pipeline changes 2 + 3).

Holds the robot up with the full harness and drives both hips fully inward, so the feet of a
narrow-stance body (trex_v2) meet. Episode resets are disabled so each env keeps pushing.
With `robot.self_collision` on, every env must report a pair force (which terminates the
episode) and the feet must stay apart; with it off, the feet pass through each other and no
pair is reported.

    python redesign/test_self_collision.py --backend sim --self-collision on
    python redesign/test_self_collision.py --backend lab --self-collision on   # isaaclab6 env
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=["sim", "lab"], default="sim")
parser.add_argument("--task", default="redesign/experiments/v2a_base.yaml")
parser.add_argument("--self-collision", choices=["on", "off"], default="on")
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--min-gap-mm", type=float, default=60.0, help="feet centre gap the collision must hold")
args = parser.parse_args()

if args.backend == "sim":
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
else:
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device="cpu").app

import torch  # noqa: E402

try:
    from walk_rl.task_core import load_task_config

    cfg = load_task_config(PROJECT_ROOT / args.task)
    cfg.robot.self_collision = args.self_collision == "on"
    cfg.domain_randomization.enabled = False
    cfg.push.max_lin_vel_mps = 0.0
    if args.backend == "sim":
        from walk_rl.env import TrexWalkEnv
        env = TrexWalkEnv(args.num_envs, cfg, device="cpu", simulation_app=app, seed=0)
    else:
        from walk_lab.env import TrexLabWalkEnv
        env = TrexLabWalkEnv(args.num_envs, cfg, device="cpu", seed=0)
    on = cfg.robot.self_collision
    print(f"[selfcol] backend={args.backend} asset={cfg.robot.asset_dir} self_collision={on} pairs={env.self_pairs}")
    env.reset()
    env.set_harness(1.0)
    env.set_push_scale(0.0)
    env.reset_idx = lambda ids: None  # keep every env pushing its feet together
    hips = [env.act_names.index(n) for n in ("left_hip_joint", "right_hip_joint")]
    a = torch.zeros(env.num_envs, env.num_actions)
    a[:, hips] = 1.0  # + = inward on either leg (docs/JOINT_CONVENTIONS.md)
    N = env.num_envs
    first_done = torch.full((N,), -1)
    peak = torch.zeros(N)
    min_gap = torch.full((N,), 1e9)
    for t in range(args.steps):
        _, _, done, _ = env.step(a)
        f = env.self_force.max(dim=1).values.cpu() if env.self_pairs else torch.zeros(N)
        peak = torch.maximum(peak, f)
        d = env.foot_pos[:, 0, :2] - env.foot_pos[:, 1, :2]
        gap = (d[:, 1].abs() * 1000).cpu()  # robot faces +x at reset: lateral = world y
        min_gap = torch.minimum(min_gap, gap)
        first_done = torch.where((first_done < 0) & done.cpu(), torch.tensor(t), first_done)
        if t % 5 == 0:
            hip = torch.rad2deg(env.q[:, env.act_idx[hips]]).mean(dim=1)
            print(f"[selfcol] t={t:2d} hip deg {hip.round(decimals=1).tolist()} gap mm {gap.round().tolist()} "
                  f"pair N {f.round(decimals=2).tolist()}")
    print(f"[selfcol] first termination step per env {first_done.tolist()}; peak pair force N {peak.round(decimals=2).tolist()}; "
          f"min feet gap mm {min_gap.round(decimals=1).tolist()}")
    if on:
        ok = bool((first_done >= 0).all() and (peak > cfg.termination.self_contact_force_n).all()
                  and (min_gap >= args.min_gap_mm).all())
    else:
        ok = bool((first_done < 0).all() and (min_gap < args.min_gap_mm).all())
    print(f"[selfcol] {'PASS' if ok else 'FAIL'}")
except Exception:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
finally:
    app.close()
