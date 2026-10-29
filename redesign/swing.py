"""Swing-leg speed test (policy-free): how fast can one leg move its foot by a step, in the air, on the
fitted LX-16A servo model? The torso is held fixed (root pinned every substep), the other leg is
held at its pose, gravity is on. Goals are sent like the Pi runner does: one new goal per 40 ms
control tick, rate-limited to the runner's max_step, servo move time = one tick.

For each variant the start/end poses are found by inverse kinematics at the same body height
(hip-to-sole = `height_frac` x the straight-leg value) on the reverse-knee branch the policy walks
on, for a foot displacement of `step_m` (back -> front). Reported: time until the foot is within
2 mm of the target position, and the peak femur/tibia torque fraction.

    python redesign/swing.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from redesign.statics import Statics  # noqa: E402
from redesign.variant_model import patched_backend  # noqa: E402
import pi_runner.mujoco_backend as mb  # noqa: E402


def sole_rel_hip(st: Statics, qf: float, qt: float) -> np.ndarray:
    st.set_pose({"left_femur_joint": qf, "left_tibia_joint": qt})
    hip = st.d.xpos[st.m.body("left_femur").id]
    c = st.sole_center_world("left") - hip
    return np.array([c[0], c[2]])


def ik(st: Statics, x: float, z: float, guess=(-15.0, -35.0)):
    r = least_squares(lambda q: (sole_rel_hip(st, q[0], q[1]) - [x, z]) * 1000, guess, bounds=([-90, -100], [90, 0]))
    return r.x, float(np.abs(r.fun).max())


def swing_time(variant: dict, step_m: float = 0.06, height_frac: float = 0.97, lift_m: float = 0.0,
               limits: dict | None = None, timeout_s: float = 1.5) -> dict:
    st = Statics(variant)
    straight = sole_rel_hip(st, 0.0, 0.0)
    H = straight[1] * height_frac
    x0 = straight[0]
    qa, ea = ik(st, x0 - step_m / 2, H)
    qb, eb = ik(st, x0 + step_m / 2, H)
    with patched_backend(variant):
        sim = mb.MujocoTrex(friction=0.5, control_hz=25, physics_hz=1000, actuator="lx16a")
    m, d = sim.model, sim.data
    names = sim.act_names
    iF, iT = names.index("left_femur_joint"), names.index("left_tibia_joint")
    # pin the torso 0.5 m up; start in pose A
    root_q = d.qpos[:7].copy()
    root_q[2] = 0.5
    start = np.zeros(8)
    start[iF], start[iT] = np.radians(qa)
    d.qpos[:7] = root_q
    d.qpos[sim.qadr] = start
    ank = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_ankle_joint")]
    d.qpos[ank] = start[iF] - start[iT]
    mujoco.mj_forward(m, d)
    sim.goal = start.copy()
    sim.queue.clear()
    sim.queue.append(start.copy())
    target = start.copy()
    target[iF], target[iT] = np.radians(qb)
    max_step = sim.rc.actuator_no_load_speed_rad_per_sec / 25.0  # runner rate limit (datasheet speed per tick)
    foot = st.foot["left"]
    from walk_rl.actuator import lx16a_torque, ramp_reference
    p, v = sim.lx, sim.lx.v_nominal / sim.lx.v_ref
    stall = p.stall * v
    goal = start.copy()
    t_reach, peak = None, 0.0
    hip_id = m.body("left_femur").id
    for k in range(int(timeout_s * 25)):
        goal = goal + np.clip(target - goal, -max_step, max_step)
        sim.queue.append(goal.copy())
        tgt, st0 = sim.queue[0], sim.goal
        for i in range(sim.decimation):
            q_ref = ramp_reference(st0, tgt, (i + 1) * m.opt.timestep, sim.control_period, p.latency, np.clip)
            tau = lx16a_torque(q_ref, d.qpos[sim.qadr], d.qvel[sim.dofadr], p.kp, p.deadband, p.coulomb, stall, p.no_load * v,
                               p.armature, m.opt.timestep, np)
            d.ctrl[sim.ctrl_idx] = tau
            peak = max(peak, abs(tau[iF]) / stall, abs(tau[iT]) / stall)
            mujoco.mj_step(m, d)
            d.qpos[:7] = root_q
            d.qvel[:6] = 0
            mujoco.mj_forward(m, d)
        sim.goal = tgt.copy()
        # foot position relative to hip (sole centre), compare with target
        b = st.foot["left"]
        R = d.xmat[b].reshape(3, 3)
        c = st.sole["left"].centroid
        sole = d.xpos[b] + R @ np.array([c.x, c.y, st.sole_z["left"]]) - d.xpos[hip_id]
        if t_reach is None and abs(sole[0] - (x0 + step_m / 2)) < 0.002 and abs(sole[2] - H) < 0.002:
            t_reach = (k + 1) / 25.0
            break
    return {"variant": variant, "step_m": step_m, "H_m": float(H), "q_start_deg": qa.round(1).tolist(),
            "q_end_deg": qb.round(1).tolist(), "ik_err_mm": max(ea, eb), "t_reach_s": t_reach, "peak_tau_frac": peak,
            "dq_deg": (qb - qa).round(1).tolist()}


VARIANTS = [
    ("baseline", {}),
    ("thigh 75 mm", {"thigh_length_m": 0.075}),
    ("thigh 85 mm", {"thigh_length_m": 0.085}),
    ("shank 64 mm", {"shank_length_m": 0.064}),
    ("thigh 75 + shank 64", {"thigh_length_m": 0.075, "shank_length_m": 0.064}),
    ("thigh 55 mm", {"thigh_length_m": 0.055}),
    ("leg mass x0.7", {"leg_mass_scale": 0.7}),
    ("leg mass x1.3", {"leg_mass_scale": 1.3}),
]

if __name__ == "__main__":
    rows = []
    for step in (0.04, 0.08):
        for label, var in VARIANTS:
            r = swing_time(var, step)
            r["label"] = label
            rows.append(r)
            print(f"step {step*1000:.0f} mm  {label:22s} H {r['H_m']*1000:6.1f} mm  dq {r['dq_deg']}  "
                  f"reach {r['t_reach_s']} s  peak tau {r['peak_tau_frac']:.2f}  ik err {r['ik_err_mm']:.2f} mm", flush=True)
    (ROOT / "outputs" / "redesign" / "swing.json").write_text(json.dumps(rows, indent=1))
