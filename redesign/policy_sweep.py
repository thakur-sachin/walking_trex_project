"""Policy-in-the-loop sweep: the deployed v9c loop in MuJoCo (fitted LX-16A model) on design variants.

Only levers the policy's training randomization covered are a fair test (see policy_probe.py); rows
outside that range are tagged "ood" and read as "needs retraining", not as design quality.

    python redesign/policy_sweep.py [--procs 8] [--seconds 30]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# (label, variant, inside the v9c training randomization?)
VARIANTS = [
    ("baseline", {}, True),
    ("torso 0.8 kg", {"base_mass_kg": 0.8}, True),
    ("torso 0.9 kg", {"base_mass_kg": 0.9}, True),
    ("torso 1.1 kg", {"base_mass_kg": 1.1}, True),
    ("torso 1.2 kg", {"base_mass_kg": 1.2}, True),
    ("torso 1.3 kg", {"base_mass_kg": 1.3}, True),
    ("torso 0.6 kg", {"base_mass_kg": 0.6}, False),
    ("torso 1.5 kg", {"base_mass_kg": 1.5}, False),
    ("head+tail x0.85", {"head_mass_scale": 0.85, "tail_mass_scale": 0.85}, True),
    ("head+tail x1.15", {"head_mass_scale": 1.15, "tail_mass_scale": 1.15}, True),
    ("head+tail x0.5", {"head_mass_scale": 0.5, "tail_mass_scale": 0.5}, False),
    ("torso COM z -15 mm", {"base_com_shift_m": [0, 0, -0.015]}, True),
    ("torso COM z +15 mm", {"base_com_shift_m": [0, 0, 0.015]}, True),
    ("torso COM x -15 mm", {"base_com_shift_m": [-0.015, 0, 0]}, True),
    ("torso COM x +15 mm", {"base_com_shift_m": [0.015, 0, 0]}, True),
    ("stall x0.8", {"stall_scale": 0.8}, True),
    ("hips 10 mm narrower", {"hip_spacing_delta_m": -0.01}, False),
    ("hips 20 mm narrower", {"hip_spacing_delta_m": -0.02}, False),
]
CMDS = [0.0, 0.15, 0.2, 0.25]
DELAYS = [0, 1, 2]
FRICTIONS = [0.5, 0.3]


def job(args):
    from redesign.policy_probe import run_episode
    label, var, inside, cmd, delay, fr, seconds = args
    r = run_episode(ROOT / "outputs" / "walk_runs" / "v9c", var, cmd, seconds, fr, delay)
    r.update(label=label, in_dr=inside)
    r.pop("series", None)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "redesign" / "policy_sweep.json"))
    a = ap.parse_args()
    jobs = [(l, v, i, c, d, f, a.seconds) for (l, v, i), c, d, f in itertools.product(VARIANTS, CMDS, DELAYS, FRICTIONS)]
    t0 = time.time()
    with Pool(a.procs) as p:
        rows = p.map(job, jobs, chunksize=1)
    Path(a.out).write_text(json.dumps(rows, indent=1, default=float))
    print(f"{len(rows)} episodes in {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
