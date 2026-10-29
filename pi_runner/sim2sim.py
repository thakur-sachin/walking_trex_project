"""Pi-runner pass test 3: the deployed loop (runner + NumPy policy) walks in MuJoCo, a physics
engine the policy never trained in, across floor friction and bus-latency sweeps.

    python pi_runner/sim2sim.py --run v8d [--seconds 60] [--noisy]

For each friction x delay x command it runs one 60 s episode and reports survival time, the
reason for any fall, forward speed (heading frame), sideways drift and heading change.
Pass criterion (plan A6): survives the full 60 s in every walking condition.
Writes outputs/walk_runs/<run>/sim2sim_<policy>.json.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from pi_runner.mujoco_backend import MujocoTrex  # noqa: E402
from pi_runner.runner import PolicyRunner  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--run", default="v8d")
parser.add_argument("--policy", default="policy_numpy.npz")
parser.add_argument("--seconds", type=float, default=60.0)
parser.add_argument("--frictions", type=float, nargs="+", default=[0.5, 0.8, 1.2])
parser.add_argument("--delays", type=int, nargs="+", default=[0, 1, 2])
parser.add_argument("--contact-tc", type=float, nargs="+", default=[0.004],
                    help="MuJoCo contact time constant(s) [s]: 0.004 near-rigid (PhysX-like), 0.01-0.02 softer floor/sole")
parser.add_argument("--commands", type=float, nargs="+", default=[0.0, 0.10, 0.20])
parser.add_argument("--noisy", action="store_true", help="training's sensor noise + encoder offsets on the readings")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--physics-hz", type=int, default=1000, help="MuJoCo step rate (see mujoco_backend: ankle linkage)")
parser.add_argument("--actuator", choices=["implicit_pd", "lx16a"], default="implicit_pd",
                    help="lx16a: the fitted LX-16A servo model (walk_rl/actuator.py), nominal parameters")
parser.add_argument("--out", default=None)
args = parser.parse_args()

run_dir = PROJECT_ROOT / "outputs" / "walk_runs" / args.run


def episode(friction: float, delay: int, cmd: float, tc: float) -> dict:
    runner = PolicyRunner(run_dir, args.policy)
    ob = runner.cfg.observation
    noise = vars(ob.noise) if args.noisy else None
    sim = MujocoTrex(friction=friction, control_hz=runner.cfg.sim.control_hz, physics_hz=args.physics_hz,
                     delay_steps=delay, contact_timeconst_s=tc, noise=noise, encoder_offset_rad=ob.encoder_offset_rad if args.noisy else 0.0,
                     seed=args.seed, max_tilt_deg=runner.cfg.termination.max_tilt_deg,
                     min_head_height_m=runner.cfg.termination.min_head_height_m,
                     ground_contact_force_n=runner.cfg.termination.ground_contact_force_n, actuator=args.actuator,
                     asset_dir=runner.asset_dir)
    runner.reset_state(np.zeros(8, dtype=np.float32), phase=0.0)
    r0 = sim.read()
    runner.latch_heading(r0.yaw)
    p0, yaw0 = sim.base_pos(), sim.yaw()
    steps = int(round(args.seconds * runner.cfg.sim.control_hz))
    vx_sum, n, res = 0.0, 0, 0.0
    for k in range(steps):
        r = sim.read()
        goal = runner.step(r, cmd)
        sim.write(goal)
        v = sim.base_vel_world()
        yaw = sim.yaw()
        vx_sum += math.cos(yaw) * v[0] + math.sin(yaw) * v[1]
        res = max(res, sim.ankle_residual_deg())
        n += 1
        if sim.failed():
            break
    p = sim.base_pos()
    d = p - p0
    fwd = (math.cos(yaw0) * d[0] + math.sin(yaw0) * d[1])
    side = (-math.sin(yaw0) * d[0] + math.cos(yaw0) * d[1])
    dyaw = math.degrees(math.atan2(math.sin(sim.yaw() - yaw0), math.cos(sim.yaw() - yaw0)))
    return {"friction": friction, "delay": delay, "contact_tc": tc, "cmd": cmd, "survived_s": round(sim.time, 2),
            "fell": sim.failed(), "vx": vx_sum / max(n, 1), "forward_m": fwd, "side_m": side, "yaw_deg": dyaw,
            "ankle_res_max_deg": res}


t0 = time.time()
rows = []
print(f"[sim2sim] run={args.run} policy={args.policy} engine=MuJoCo seconds={args.seconds} noisy={args.noisy}")
print("[sim2sim] tc     fric delay  cmd | survived      | vx (%cmd)       | forward | side   | yaw    | ankle res max")
for tc in args.contact_tc:
  for fr in args.frictions:
    for dl in args.delays:
        for c in args.commands:
            r = episode(fr, dl, c, tc)
            rows.append(r)
            pct = f"({100 * r['vx'] / c:4.0f}%)" if c > 0 else "       "
            surv = f"{r['survived_s']:5.1f}s" + (f" {r['fell']}" if r["fell"] else " ok")
            print(f"[sim2sim] {tc:.3f}  {fr:.1f}  {dl}   {c:.2f} | {surv:13s} | {r['vx']:+.3f} {pct} | {r['forward_m']:+6.2f}m | "
                  f"{r['side_m']:+.2f}m | {r['yaw_deg']:+6.1f} | {r['ankle_res_max_deg']:.2f} deg", flush=True)
walking = [r for r in rows if r["cmd"] > 0 and r["contact_tc"] == min(args.contact_tc)]
full = sum(r["fell"] is None for r in rows)
ok = all(r["fell"] is None for r in walking)
print(f"[sim2sim] survived the full {args.seconds:.0f} s: {full}/{len(rows)} episodes; "
      f"{'OK' if ok else 'FAILED'} (pass: every walking condition on the stiffest floor survives) [{time.time() - t0:.0f}s wall]")
out = Path(args.out) if args.out else run_dir / f"sim2sim_{Path(args.policy).stem}{'_noisy' if args.noisy else ''}{'_lx16a' if args.actuator == 'lx16a' else ''}.json"
out.write_text(json.dumps({"run": args.run, "policy": args.policy, "seconds": args.seconds, "noisy": args.noisy,
                           "rows": rows, "pass": ok}, indent=1))
print(f"[sim2sim] wrote {out}")
