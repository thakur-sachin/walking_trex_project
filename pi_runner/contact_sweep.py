"""Which MuJoCo contact and joint settings make the policy walk like the real robot?

    $LAB pi_runner/contact_sweep.py assets/robots/trex/servo_logs/walk_v9o_20261021_212933_floor.npz \
        --run v9o --real-distance 0.46

The policy runs closed loop in the deployed loop (pi_runner/runner.py) with the log's command schedule,
once per setting. Each walk is reduced to the gait features that differ between the real floor and the
nominal MuJoCo walk (2026-10-21, v9o): mean and swing (std) of the hips, the right femur mean, the tibia
swings, the mean roll while walking, and the speed. The score is the RMS of the feature errors against
the real log, each scaled by its tolerance (FEATURES). Lower is closer to the real robot.

Settings swept (each overrides the nominal model in memory; nothing is written to robot_config):
  mu        sliding friction (floor and robot geoms)
  torsion   torsional friction; no effect with the model's condim 3 contacts (kept for a condim 4/6 model)
  tc        contact time constant in s (softer floor or sole)
  deadband  servo deadband in deg (a stand-in for joint backlash; nominal 0.41)

Result 2026-10-21 (v9o log): a low friction (mu 0.15: 47 %) or 3 deg of play (50-57 %) brings the speed down
to the real 46 %, but no symmetric contact or joint setting gives the real one-sided posture (left hip -6.5
deg with a +-6 deg swing, right femur -5 deg). One-sided encoder offsets (+-4 deg) and one-sided play (4 deg)
on the hips and right leg do not either: the policy compensates them. Best score 2.2 (1 = within tolerance).
"""
import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pi_runner.mujoco_backend import MujocoTrex  # noqa: E402
from pi_runner.replay_real_walk import load_log  # noqa: E402
from pi_runner.runner import PolicyRunner  # noqa: E402

# name -> tolerance used to scale the error (deg, or % of the command for speed)
FEATURES = {"lhip_mean": 1.5, "lhip_std": 1.0, "rhip_mean": 1.5, "rhip_std": 1.0, "rfem_mean": 1.5,
            "ltib_std": 1.0, "rtib_std": 1.0, "roll_mean": 0.5, "speed_pct": 5.0}


def roll_pitch_deg(g: np.ndarray):
    g = g / np.linalg.norm(g, axis=1, keepdims=True)
    return np.degrees(np.arctan2(-g[:, 1], -g[:, 2])), np.degrees(np.arcsin(np.clip(g[:, 0], -1, 1)))


def features(q: np.ndarray, gravity: np.ndarray, walk: np.ndarray, forward_m: float, seconds: float, cmd: float) -> dict:
    qd = np.degrees(q[walk])
    roll, _ = roll_pitch_deg(gravity[walk])
    return {"lhip_mean": qd[:, 2].mean(), "lhip_std": qd[:, 2].std(), "rhip_mean": qd[:, 3].mean(),
            "rhip_std": qd[:, 3].std(), "rfem_mean": qd[:, 5].mean(), "ltib_std": qd[:, 6].std(),
            "rtib_std": qd[:, 7].std(), "roll_mean": roll.mean(), "speed_pct": forward_m / seconds / cmd * 100}


def score(f: dict, ref: dict) -> float:
    return math.sqrt(np.mean([((f[k] - ref[k]) / tol) ** 2 for k, tol in FEATURES.items()]))


def walk_sim(run_dir: Path, log: dict, walk: np.ndarray, mu: float, torsion: float, tc: float, deadband_deg: float):
    runner = PolicyRunner(run_dir)
    sim = MujocoTrex(friction=mu, contact_timeconst_s=tc, actuator="lx16a",
                     max_tilt_deg=runner.cfg.termination.max_tilt_deg,
                     min_head_height_m=runner.cfg.termination.min_head_height_m)
    sim.model.geom_friction[:, 1] = torsion
    sim.lx.deadband = math.radians(deadband_deg)
    runner.reset_state(np.zeros(8, dtype=np.float32), phase=0.0)
    runner.latch_heading(sim.read().yaw)
    Q, G, fwd = [], [], 0.0
    for i in range(len(log["t"])):
        r = sim.read()
        Q.append(r.q); G.append(r.gravity_b)
        p0, h = sim.base_pos(), sim.yaw()
        sim.write(runner.step(r, float(log["cmd"][i])))
        if walk[i]:
            d = sim.base_pos() - p0
            fwd += math.cos(h) * d[0] + math.sin(h) * d[1]
        if sim.failed():
            return None, f"fell ({sim.failed()}) at {log['t'][i]:.1f} s"
    return (np.array(Q), np.array(G), fwd), None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("--run", required=True)
    ap.add_argument("--real-distance", type=float, required=True, help="measured distance while walking, m")
    ap.add_argument("--mu", type=float, nargs="+", default=[0.15, 0.2, 0.3, 0.5, 0.8])
    ap.add_argument("--torsion", type=float, nargs="+", default=[0.005])
    ap.add_argument("--tc", type=float, nargs="+", default=[0.004, 0.01, 0.02])
    ap.add_argument("--deadband", type=float, nargs="+", default=[0.41, 1.5, 3.0])
    ap.add_argument("--from-t", type=float, default=4.0, help="ignore the first seconds of the walk (start-up)")
    ap.add_argument("--json", type=Path, help="write all results here")
    a = ap.parse_args()

    log = load_log(a.log)
    walk = (log["cmd"] > 0) & (log["t"] > a.from_t)
    cmd = float(log["cmd"][walk].max())
    seconds = walk.sum() * 0.04
    move = log["cmd"] > 0  # the pens measure the whole walk; scale it to the analysed window
    ref = features(log["q"], log["gravity"], walk, a.real_distance * seconds / (move.sum() * 0.04), seconds, cmd)
    keys = list(FEATURES)
    head = f"{'mu':>5s} {'tors':>6s} {'tc':>6s} {'db':>5s} | " + " ".join(f"{k:>9s}" for k in keys) + " | score"
    print(head)
    print(f"{'real':>26s} | " + " ".join(f"{ref[k]:+9.1f}" for k in keys))
    rows = []
    for mu, tor, tc, db in itertools.product(a.mu, a.torsion, a.tc, a.deadband):
        out, fell = walk_sim(PROJECT_ROOT / "outputs" / "walk_runs" / a.run, log, walk, mu, tor, tc, db)
        tag = f"{mu:5.2f} {tor:6.3f} {tc:6.3f} {db:5.2f}"
        if fell:
            print(f"{tag} | {fell}")
            rows.append({"mu": mu, "torsion": tor, "tc": tc, "deadband": db, "fell": fell})
            continue
        f = features(out[0], out[1], walk, out[2], seconds, cmd)
        s = score(f, ref)
        print(f"{tag} | " + " ".join(f"{f[k]:+9.1f}" for k in keys) + f" | {s:5.2f}", flush=True)
        rows.append({"mu": mu, "torsion": tor, "tc": tc, "deadband": db, "score": s, **{k: float(v) for k, v in f.items()}})
    ok = sorted((r for r in rows if "score" in r), key=lambda r: r["score"])
    print("\nclosest to the real walk:")
    for r in ok[:8]:
        print(f"  mu {r['mu']:.2f} torsion {r['torsion']:.3f} tc {r['tc']:.3f} deadband {r['deadband']:.2f}"
              f" -> score {r['score']:.2f}, speed {r['speed_pct']:.0f}%")
    if a.json:
        a.json.write_text(json.dumps({"real": {k: float(v) for k, v in ref.items()}, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
