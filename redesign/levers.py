"""Static / quasi-static figures of merit for one design variant, and the one-at-a-time lever sweep
defined in redesign/studies.yaml.

    python redesign/levers.py                       # all sweeps -> outputs/redesign/levers.json + table
    python redesign/levers.py --variant '{"hip_spacing_delta_m": -0.03}'

Figures of merit (all quasi-static; see redesign/statics.py for the method):
  mass_kg, com_h_m           total mass, centre of mass height above the soles (walking pose)
  halfwidth_m                half the distance between the two sole centres
  ss_hip_switch              stance hip roll torque / stall, single support, LIPM step switch
                             (centre of mass at the midline, ground force through it)
  ss_hip_mid                 the same at LIPM mid-stance: x 1/cosh(omega T/2) (linear scaling)
  ss_mu_switch               friction coefficient the step switch needs (|F_xy| / F_z)
  ss_femur, ss_tibia         stance femur / tibia torque / stall at the walking pose, CoP under
                             the centre of mass and +-20 mm along the foot (max |.|)
  ds_stand_max               largest servo torque / stall standing in the v9c standing pose
  sway_m, roll_est_deg       LIPM lateral sway of the centre of mass and the torso roll that would
                             produce it pivoting about the stance foot (half amplitude)
  t_pitch_s, t_roll_s        inverted-pendulum time constant of a fall over the toe / outer sole edge
  foot_mass_x_kg, foot_mass_z_kg   swing-foot effective mass (torso fixed), servo armature included
  foot_vx_max_mps            swing-foot speed from femur + tibia at the fitted no-load speed
  stance_fz_max_n            vertical foot force femur + tibia can hold at the walking pose (stall)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from redesign.statics import POSES, Statics, G  # noqa: E402

STUDIES = ROOT / "redesign" / "studies.yaml"


def foot_space(st: Statics, side: str, pose: dict):
    """Swing-foot effective mass (operational space, torso fixed) and speed / force capability."""
    m, d = st.m, st.d
    st.set_pose(pose)
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, M, d.qM)
    dofs = [st.dof[f"{side}_femur_joint"], st.dof[f"{side}_tibia_joint"]]
    an = st.dof[f"{side}_ankle_joint"]
    # ankle follows the linkage: qa = qf - qt -> generalized coordinates (qf, qt) with E mapping
    E = np.zeros((m.nv, 2))
    E[dofs[0], 0] = 1.0
    E[dofs[1], 1] = 1.0
    E[an, 0] = 1.0
    E[an, 1] = -1.0
    Mr = E.T @ M @ E
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jp, jr, st.sole_center_world(side), st.foot[side])
    J = (jp @ E)[[0, 2], :]  # foot x, z w.r.t. (qf, qt)
    Lam = np.linalg.inv(J @ np.linalg.inv(Mr) @ J.T)
    am = st.rc.raw["actuator_model"]
    w0 = st.rc.actuator_no_load_speed_rad_per_sec * am["no_load_speed_scale"] * am["measured_voltage_v"] / am["reference_voltage_v"]
    vx_max = w0 * np.abs(J[0]).sum()
    # vertical force: tau = J^T F (servo side; foot moments go through the linkage)
    fz_max = st.stall / np.abs(J[1]).max()
    return float(Lam[0, 0]), float(Lam[1, 1]), float(vx_max), float(fz_max)


def figures(variant: dict, gait_hz: float = 2.2, asset_dir=None) -> dict:
    st = Statics(variant, asset_dir)
    out = {"mass_kg": st.total_mass()}
    walk, stand = POSES["v9c_walk"], POSES["v9c_stand"]
    st.set_pose(walk)
    h = st.com_height_above_sole()
    out["com_h_m"] = h
    out["halfwidth_m"] = st.stance_halfwidth()
    I = st.inertia_about_com()
    m = st.total_mass()
    # single support, step switch
    ss = st.single_support(walk, "left")
    out["ss_hip_switch"] = abs(ss.tau_frac["left_hip_joint"])
    out["ss_mu_switch"] = ss.friction_needed
    om = math.sqrt(G / h)
    T = 1.0 / (2.0 * gait_hz)
    k = 1.0 / math.cosh(om * T / 2)
    out["ss_hip_mid"] = out["ss_hip_switch"] * k
    out["sway_m"] = out["halfwidth_m"] * (1 - k)
    out["roll_est_deg"] = math.degrees(math.atan2(out["sway_m"], h))
    # femur / tibia in stance, CoP under the CoM and +-20 mm along the foot
    st.set_pose(walk)
    c = st.com()
    sc = st.sole_center_world("left")
    vals = {"left_femur_joint": [], "left_tibia_joint": []}
    for dx in (-0.02, 0.0, 0.02):
        r = st.single_support(walk, "left", cop_offset=(c[0] - sc[0] + dx, 0.0))
        for j in vals:
            vals[j].append(abs(r.tau_frac[j]))
    out["ss_femur"] = max(vals["left_femur_joint"])
    out["ss_tibia"] = max(vals["left_tibia_joint"])
    out["ss_femur_tibia_at_com"] = [vals["left_femur_joint"][1], vals["left_tibia_joint"][1]]
    ds = st.double_support(stand)
    out["ds_stand_max"] = max(abs(v) for k_, v in ds.tau_frac.items() if "hip" in k_ or "femur" in k_ or "tibia" in k_)
    out["ds_stand_headtail"] = [abs(ds.tau_frac["head_joint"]), abs(ds.tau_frac["tail_joint"])]
    out["ds_support_margin_m"] = ds.margins["support_margin_m"]
    # pendulum time constants: pivot at the front sole edge (pitch) and outer sole edge (roll)
    st.set_pose(walk)
    poly, zs = st.sole_polygon_world("left")
    x_front = max(st.sole_polygon_world(s)[0].bounds[2] for s in ("left", "right"))
    y_out = st.sole_polygon_world("left")[0].bounds[3]
    c = st.com()
    dxp, dyr = x_front - c[0], y_out - c[1]
    Ip = I[1, 1] + m * (h * h + dxp * dxp)
    Ir = I[0, 0] + m * (h * h + dyr * dyr)
    out["t_pitch_s"] = math.sqrt(Ip / (m * G * h))
    out["t_roll_s"] = math.sqrt(Ir / (m * G * h))
    out["I_com_diag"] = np.diag(I).tolist()
    # single support, static: signed distance of the centre of mass from the stance sole's convex
    # hull (positive inside). Negative = the body must lean / fall toward the stance foot to stand on it.
    import shapely
    out["ss_static_margin_m"] = -float(shapely.Point(c[0], c[1]).distance(poly)) if not poly.contains(shapely.Point(c[0], c[1])) \
        else float(poly.exterior.distance(shapely.Point(c[0], c[1])))
    out["sole_inner_gap_m"] = float(st.sole_polygon_world("left")[0].bounds[1] - st.sole_polygon_world("right")[0].bounds[3])
    fx, fz, vx, fzmax = foot_space(st, "left", walk)
    out.update(foot_mass_x_kg=fx, foot_mass_z_kg=fz, foot_vx_max_mps=vx, stance_fz_max_n=fzmax,
               stance_fz_need_n=m * G)
    return out


def run_sweeps(cfg: dict) -> dict:
    res = {"baseline": figures({})}
    for name, sw in cfg["sweeps"].items():
        rows = []
        for val in sw["values"]:
            var = copy.deepcopy(sw.get("fixed", {}))
            if sw.get("key"):
                var[sw["key"]] = val
            else:
                var.update(val)
            rows.append({"value": val, "variant": var, **figures(var)})
        res[name] = rows
    return res


COLS = ["ss_static_margin_m", "sole_inner_gap_m", "mass_kg", "com_h_m", "halfwidth_m", "ss_hip_switch", "ss_hip_mid", "ss_mu_switch", "ss_femur", "ss_tibia",
        "ds_stand_max", "roll_est_deg", "t_pitch_s", "t_roll_s", "foot_mass_x_kg", "foot_vx_max_mps", "stance_fz_max_n"]


def fmt_row(label, f):
    return f"{label:34s} " + " ".join(f"{f[c]:8.3f}" for c in COLS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "redesign" / "levers.json"))
    a = ap.parse_args()
    if a.variant:
        print(json.dumps(figures(json.loads(a.variant)), indent=1))
        return
    cfg = yaml.safe_load(STUDIES.read_text())
    res = run_sweeps(cfg)
    print(f"{'':34s} " + " ".join(f"{c[:8]:>8s}" for c in COLS))
    print(fmt_row("baseline", res["baseline"]))
    for name, rows in res.items():
        if name == "baseline":
            continue
        for r in rows:
            print(fmt_row(f"{name}={r['value']}"[:34], r))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
