"""v9c in MuJoCo with the as-built mechanical stops added as hard joint limits (policy unchanged):
ankle (= femur - tibia) -57..+39 deg from the foot~lower-idler / foot~shank contacts, hip inward +10 deg
(hip bracket ~ hip servo). Answers: does the current policy run into stops the robot has?

    python redesign/asbuilt_stops_probe.py [--procs 8]
"""
import argparse
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ANK = {"left_ankle_joint": [-57, 39], "right_ankle_joint": [-57, 39]}
HIP = {"left_hip_joint": [-30, 10], "right_hip_joint": [-30, 10]}
VARIANTS = [("model (no stops)", {}), ("ankle stops", {"joint_limits_deg": ANK}),
            ("ankle + hip stops", {"joint_limits_deg": {**ANK, **HIP}})]


def job(a):
    from redesign.policy_probe import run_episode
    label, var, cmd, delay, fr = a
    r = run_episode(ROOT / "outputs" / "walk_runs" / "v9c", var, cmd, 30.0, fr, delay)
    r.pop("series", None)
    r["label"] = label
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    jobs = [(l, v, c, d, f) for (l, v), c, d, f in itertools.product(VARIANTS, [0.0, 0.15, 0.2, 0.25], [0, 1, 2], [0.5, 0.3])]
    with Pool(a.procs) as p:
        rows = p.map(job, jobs, chunksize=1)
    (ROOT / "outputs" / "redesign" / "asbuilt_stops_probe.json").write_text(json.dumps(rows, indent=1, default=float))
    for label, _ in VARIANTS:
        for fr in (0.5, 0.3):
            rs = [r for r in rows if r["label"] == label and r["friction"] == fr]
            w = [r for r in rs if r["cmd"] > 0]
            vx = {c: sum(r.get("vx", 0) for r in w if r["cmd"] == c) / 3 for c in (0.15, 0.2, 0.25)}
            print(f"{label:20s} mu {fr}: walking survival {sum(r['fell'] is None for r in w)}/{len(w)}, standing "
                  f"{sum(r['fell'] is None for r in rs if r['cmd'] == 0)}/3, vx " + " ".join(f"{v:.3f}" for v in vx.values())
                  + f", falls: {[r['fell'] for r in rs if r['fell']]}")
