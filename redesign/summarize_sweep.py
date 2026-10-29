"""Table of outputs/redesign/policy_sweep.json: per variant, walking commands pooled over delays."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
rows = json.loads((ROOT / "outputs" / "redesign" / (sys.argv[1] if len(sys.argv) > 1 else "policy_sweep.json")).read_text())
by = defaultdict(list)
for r in rows:
    by[(r["label"], r["friction"])].append(r)
hdr = f"{'variant':24s} {'DR':3s} {'mu':>4s} | {'surv':>5s} | {'vx@.15':>6s} {'vx@.20':>6s} {'vx@.25':>6s} | {'roll':>5s} | {'hip>80':>6s} {'fem>80':>6s} {'tib>80':>6s} | {'hip':>5s} {'fem':>5s} {'tib':>5s} | {'slip':>5s} | stand surv"
print(hdr)
for (label, fr), rs in by.items():
    walk = [r for r in rs if r["cmd"] > 0]
    stand = [r for r in rs if r["cmd"] == 0]
    ok = [r for r in walk if r["fell"] is None]
    surv = len(ok) / len(walk)
    vx = {c: np.mean([r["vx"] for r in walk if r["cmd"] == c and "vx" in r]) for c in (0.15, 0.2, 0.25)}
    g = lambda k: np.mean([r[k] for r in walk if k in r])
    print(f"{label:24s} {'in' if rs[0]['in_dr'] else 'OOD':3s} {fr:4.1f} | {surv:5.2f} | {vx[0.15]:6.3f} {vx[0.2]:6.3f} {vx[0.25]:6.3f} | "
          f"{g('roll_p2p_deg'):5.1f} | {g('hip_over80'):6.3f} {g('femur_over80'):6.3f} {g('tibia_over80'):6.3f} | "
          f"{g('hip_mean_frac'):5.2f} {g('femur_mean_frac'):5.2f} {g('tibia_mean_frac'):5.2f} | {g('foot_slip_mps'):5.3f} | "
          f"{sum(r['fell'] is None for r in stand)}/{len(stand)}")
