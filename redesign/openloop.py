"""Policy-free, open-loop MuJoCo tests of design variants on the fitted LX-16A servo model.

Each test sends a fixed schedule of servo goals at the 25 Hz control rate (move-time ramps, as the
Pi does) and measures what the body does. No feedback: these tests probe what the MECHANISM allows
(standing without control, standing on one foot), not how well a policy could walk.

  stand        hold a pose 10 s
  foot_lift    from the zero pose lift one foot ~10 mm (knee bend, femur +25 / tibia +45 deg) for
               2 s, put it down, then the other foot. Reports how long each lifted foot was really
               off the ground (contact force < 1 N), the torso roll, and falls.

    python redesign/openloop.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from redesign.variant_model import patched_backend  # noqa: E402
import pi_runner.mujoco_backend as mb  # noqa: E402
from redesign.policy_probe import roll_pitch  # noqa: E402

J = ["head_joint", "tail_joint", "left_hip_joint", "right_hip_joint", "left_femur_joint", "right_femur_joint",
     "left_tibia_joint", "right_tibia_joint"]


def pose_vec(p: dict) -> np.ndarray:
    return np.radians([p.get(n, 0.0) for n in J])


def lerp(a, b, s):
    return a + (b - a) * min(max(s, 0.0), 1.0)


def foot_lift_schedule(t: float, lift: dict) -> np.ndarray:
    z = pose_vec({})
    R = pose_vec({"right_femur_joint": lift["femur"], "right_tibia_joint": lift["tibia"]})
    L = pose_vec({"left_femur_joint": lift["femur"], "left_tibia_joint": lift["tibia"]})
    seq = [(1.0, z, z), (0.5, z, R), (2.0, R, R), (0.5, R, z), (1.0, z, z), (0.5, z, L), (2.0, L, L), (0.5, L, z), (1.0, z, z)]
    t0 = 0.0
    for dur, a, b in seq:
        if t < t0 + dur:
            return lerp(a, b, (t - t0) / dur)
        t0 += dur
    return z


def run(variant: dict, test: str, seconds: float | None = None, friction: float = 0.5, pose: dict | None = None,
        lift: dict | None = None) -> dict:
    with patched_backend(variant):
        sim = mb.MujocoTrex(friction=friction, control_hz=25, physics_hz=1000, actuator="lx16a")
    m, d = sim.model, sim.data
    feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_foot") for s in ("left", "right")]
    lift = lift or {"femur": 25.0, "tibia": 45.0}
    if test == "stand":
        seconds = seconds or 10.0
        goal_at = lambda t: pose_vec(pose or {})
        # start already in the pose (servos hold it from t=0)
        if pose:
            for n, v in (pose or {}).items():
                d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]] = math.radians(v)
            for s in ("left", "right"):
                f, tb = (pose or {}).get(f"{s}_femur_joint", 0.0), (pose or {}).get(f"{s}_tibia_joint", 0.0)
                d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_ankle_joint")]] = math.radians(f - tb)
            mujoco.mj_forward(m, d)
            sim.goal = pose_vec(pose)
            sim.queue.clear()
            sim.queue.append(sim.goal.copy())
    else:
        seconds = seconds or 9.0
        goal_at = lambda t: foot_lift_schedule(t, lift)
    steps = int(seconds * 25)
    rolls, off = [], {"left": 0.0, "right": 0.0}
    fell = None
    for k in range(steps):
        sim.write(goal_at((k + 1) / 25.0))
        ro, pi = roll_pitch(sim._rot())
        rolls.append(math.degrees(ro))
        cf = np.zeros(2)
        for ci in range(d.ncon):
            c = d.contact[ci]
            for fi, f in enumerate(feet):
                if m.geom_bodyid[c.geom1] == f or m.geom_bodyid[c.geom2] == f:
                    fr = np.zeros(6)
                    mujoco.mj_contactForce(m, d, ci, fr)
                    cf[fi] += abs(fr[0])
        for fi, s in enumerate(("left", "right")):
            if cf[fi] < 1.0 and cf[1 - fi] > 1.0:
                off[s] += 1 / 25.0
        if sim.failed():
            fell = sim.failed()
            break
    return {"variant": variant, "test": test, "survived_s": round(sim.time, 2), "fell": fell,
            "roll_max_abs_deg": float(np.max(np.abs(rolls))), "single_support_s": {k: round(v, 2) for k, v in off.items()}}


VARIANTS = [
    ("baseline", {}),
    ("torso 0.7 kg", {"base_mass_kg": 0.7}),
    ("head+tail x0.5", {"head_mass_scale": 0.5, "tail_mass_scale": 0.5}),
    ("hips -20 mm", {"hip_spacing_delta_m": -0.02}),
    ("hips -30 mm, feet y x0.85", {"hip_spacing_delta_m": -0.03, "foot_scale": [1.0, 0.85]}),
    ("hips -35 mm, feet y x0.75", {"hip_spacing_delta_m": -0.035, "foot_scale": [1.0, 0.75]}),
    ("foot width x1.2", {"foot_scale": [1.0, 1.2]}),
]

if __name__ == "__main__":
    from redesign.statics import POSES
    rows = []
    for label, var in VARIANTS:
        for test, kw in [("stand", {"pose": {}}), ("stand", {"pose": POSES["v9c_stand"]}), ("foot_lift", {})]:
            r = run(var, test, **kw)
            r["label"] = label
            r["pose"] = "zero" if test == "stand" and not kw.get("pose") else ("v9c_stand" if test == "stand" else "-")
            rows.append(r)
            print(f"{label:28s} {test:9s} {r['pose']:9s} survived {r['survived_s']:5.1f}s {str(r['fell']):22s} "
                  f"max|roll| {r['roll_max_abs_deg']:5.1f}  single support L/R {r['single_support_s']}", flush=True)
    (ROOT / "outputs" / "redesign" / "openloop.json").write_text(json.dumps(rows, indent=1))
