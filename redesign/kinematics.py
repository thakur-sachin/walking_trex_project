"""Sagittal leg kinematics of a design variant: reachable sole-centre positions relative to the hip
roll axis (torso level) under joint-range sets, the double-parallelogram angles, stride available at
each body height, and a workspace figure.

Parallelogram angles (the 4-bar idler links are not in the URDF; geometry from robot_config
ankle_linkage, 50 mm idler offset, perpendicular at the zero pose):
  upper (thigh, 65 mm span):  angle = femur
  lower (shank, 54 mm span):  angle = femur - tibia   (= the ankle joint angle)
A parallelogram with a perpendicular idler goes flat (singular, change point) at 90 deg; the bar
force for a given foot moment grows as 1/cos(angle). The usable limit is an ASSUMPTION
(`--para-max`, default 60 deg: bar force x2); the real one depends on idler/horn clearance.

    python redesign/kinematics.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from redesign.statics import Statics  # noqa: E402

RANGE_SETS = {  # deg: (femur lo, hi), (tibia lo, hi), optional (ankle = femur - tibia lo, hi)
    "current +-45 / +-45": ((-45, 45), (-45, 45), None),
    "tibia -70..+45": ((-45, 45), (-70, 45), None),
    "tibia -60..+60, femur +-50": ((-50, 50), (-60, 60), None),
    # as-built CAD contacts (redesign/asbuilt_ranges.py): femur -60..+63, foot~shank at ankle +39,
    # foot~lower idler at ankle -57
    "config box + as-built contacts": ((-45, 45), (-45, 45), (-57, 39)),
    "as-built contacts only": ((-60, 63), (-100, 100), (-57, 39)),
    "concept: tibia -70..+45, ankle to +60": ((-50, 50), (-70, 45), (-57, 60)),
    # current robot, no new parts: CAD contacts with 2 deg margin, femur within the left servo's travel
    "current robot extended: tibia -40..+90": ((-45, 55), (-40, 90), (-55, 37)),
    "current robot extended: tibia -40..+75": ((-45, 55), (-40, 75), (-55, 37)),
}


def workspace(st: Statics, step: float = 1.0, fem=(-90, 90), tib=(-100, 100)):
    side = "left"
    hip = None
    pts = []
    for qf in np.arange(fem[0], fem[1] + 1e-9, step):
        for qt in np.arange(tib[0], tib[1] + 1e-9, step):
            st.set_pose({f"{side}_femur_joint": qf, f"{side}_tibia_joint": qt})
            if hip is None:
                b = st.m.body(f"{side}_femur").id
                hip = st.d.xpos[b].copy()
            c = st.sole_center_world(side) - hip
            pts.append((qf, qt, c[0], c[2]))
    return np.array(pts)


def stride_at_heights(ws: np.ndarray, mask: np.ndarray, heights_mm, tol_mm: float = 1.5):
    out = {}
    for H in heights_mm:
        sel = mask & (np.abs(-ws[:, 3] * 1000 - H) < tol_mm)
        out[H] = (float(ws[sel, 2].min() * 1000), float(ws[sel, 2].max() * 1000)) if sel.any() else None
    return out


def midswing_clearance(ws: np.ndarray, mask: np.ndarray, x0_mm: float, heights_mm, tol_mm: float = 3.0):
    """Swing-foot lift available with the sole directly below its neutral position (x0), for a
    stance leg holding the body at hip-to-sole height H: H - (shortest hip-to-sole distance the
    swing leg reaches at that x). The leg is level-footed by the linkage, so only knee bend lifts it."""
    sel = mask & (np.abs(ws[:, 2] * 1000 - x0_mm) < tol_mm)
    zmin = float(-ws[sel, 3].max() * 1000) if sel.any() else None
    return {H: (None if zmin is None else max(0.0, H - zmin)) for H in heights_mm}, zmin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="{}")
    ap.add_argument("--para-max", type=float, default=60.0)
    ap.add_argument("--tag", default="baseline")
    a = ap.parse_args()
    st = Statics(json.loads(a.variant))
    ws = workspace(st)
    qf, qt = ws[:, 0], ws[:, 1]
    para_ok = (np.abs(qf) <= a.para_max) & (np.abs(qf - qt) <= a.para_max)
    hmax = float(-ws[:, 3].min() * 1000)
    hstraight = float(-ws[(qf == 0) & (qt == 0), 3][0] * 1000)
    heights = [150, 160, 170, 180, 185, 190, 195]
    x0 = float(ws[(qf == 0) & (qt == 0), 2][0] * 1000)
    res_x0 = x0
    res = {"tag": a.tag, "para_max_deg": a.para_max, "hip_to_sole_straight_mm": hstraight, "neutral_sole_x_mm": x0, "range_sets": {}}
    fig, ax = plt.subplots(1, len(RANGE_SETS), figsize=(6 * len(RANGE_SETS), 6), sharey=True)
    for i, (name, ((f0, f1), (t0, t1), ank)) in enumerate(RANGE_SETS.items()):
        box = (qf >= f0) & (qf <= f1) & (qt >= t0) & (qt <= t1)
        if ank is not None:
            box &= (qf - qt >= ank[0]) & (qf - qt <= ank[1])
        both = box & para_ok
        res["range_sets"][name] = {
            "box_only": stride_at_heights(ws, box, heights),
            "box_and_parallelogram": stride_at_heights(ws, both, heights),
            "max_parallelogram_angle_in_box_deg": float(max(np.abs(qf[box]).max(), np.abs(qf[box] - qt[box]).max())),
            "deepest_crouch_mm": float(-ws[both, 3].max() * 1000),
        }
        cl, zmin = midswing_clearance(ws, both, x0, heights)
        res["range_sets"][name]["midswing_clearance_mm"] = cl
        res["range_sets"][name]["shortest_leg_at_x0_mm"] = zmin
        # body height range at which the stance foot can sit at x0 at all
        sel = both & (np.abs(ws[:, 2] * 1000 - x0) < 3.0)
        res["range_sets"][name]["stance_height_range_at_x0_mm"] = [float(-ws[sel, 3].max() * 1000), float(-ws[sel, 3].min() * 1000)]
        ax[i].scatter(ws[box, 2] * 1000, ws[box, 3] * 1000, s=1, c="#d9a3a3", label="joint box, parallelogram > limit")
        ax[i].scatter(ws[both, 2] * 1000, ws[both, 3] * 1000, s=1, c="#4a78b5", label=f"joint box and parallelogram <= {a.para_max:.0f} deg")
        ax[i].set_title(name)
        ax[i].set_xlabel("sole centre x rel. hip roll axis [mm]")
        ax[i].set_aspect("equal")
        ax[i].grid(alpha=0.3)
    ax[0].set_ylabel("sole z rel. hip roll axis [mm]")
    ax[0].legend(loc="lower left", markerscale=8, fontsize=8)
    out = ROOT / "outputs" / "redesign"
    fig.tight_layout()
    fig.savefig(out / f"workspace_{a.tag}.png", dpi=80)
    (out / f"kinematics_{a.tag}.json").write_text(json.dumps(res, indent=1))
    for n, r in res["range_sets"].items():
        print(f"{n:28s} max parallelogram angle in box {r['max_parallelogram_angle_in_box_deg']:.0f} deg | "
              f"stance height at x0 {r['stance_height_range_at_x0_mm'][0]:.0f}-{r['stance_height_range_at_x0_mm'][1]:.0f} mm | "
              f"mid-swing lift @H " + ", ".join(f"{H}:{v:.0f}" for H, v in r['midswing_clearance_mm'].items() if v is not None) +
              " | stride @190 " + str([round(v) for v in (r['box_and_parallelogram'][190] or [])]))
    print("straight-leg hip-to-sole", round(hstraight, 1), "mm; neutral sole x", round(x0, 1), "mm")


if __name__ == "__main__":
    main()
