"""Run the deployed policy loop (PolicyRunner + NumPy policy, exactly as on the Pi) in MuJoCo on a
design variant with the fitted LX-16A servo model, and measure what the design changes.

The policy was trained on the CURRENT robot. It is only a fair probe for levers its training
randomization covered (torso mass x0.8-1.3, torso COM +-1.5 cm, head/tail mass +-15 %, friction
0.30-0.85 effective). Outside that, a drop in performance says "needs retraining", not "worse design".

    python redesign/policy_probe.py --run v9c --cmd 0.2 --seconds 20 [--variant '{"base_mass_kg": 1.2}']
"""

from __future__ import annotations

import argparse
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
from pi_runner.runner import PolicyRunner  # noqa: E402

LEG_GROUPS = {"hip": ["left_hip_joint", "right_hip_joint"], "femur": ["left_femur_joint", "right_femur_joint"],
              "tibia": ["left_tibia_joint", "right_tibia_joint"], "head": ["head_joint"], "tail": ["tail_joint"]}


def roll_pitch(R: np.ndarray) -> tuple[float, float]:
    g = R.T @ np.array([0.0, 0.0, -1.0])  # gravity in body frame
    roll = math.atan2(-g[1], -g[2])  # left side down -> g_y > 0 -> roll < 0 here; sign irrelevant for p2p
    pitch = math.atan2(g[0], math.hypot(g[1], g[2]))
    return roll, pitch


def run_episode(run_dir: Path, variant: dict | None, cmd: float, seconds: float = 20.0, friction: float = 0.5,
                delay: int = 0, warmup_s: float = 4.0, policy: str = "policy_numpy.npz", keep_series: bool = False) -> dict:
    with patched_backend(variant):
        runner = PolicyRunner(run_dir, policy)
        tc = runner.cfg.termination
        sim = mb.MujocoTrex(friction=friction, control_hz=runner.cfg.sim.control_hz, physics_hz=1000, delay_steps=delay,
                            max_tilt_deg=tc.max_tilt_deg, min_head_height_m=tc.min_head_height_m,
                            ground_contact_force_n=tc.ground_contact_force_n, actuator="lx16a")
    m, d = sim.model, sim.data
    stall_v = sim.lx.stall * sim.lx.v_nominal / sim.lx.v_ref
    names = sim.act_names
    lim = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)][1] for n in names])
    feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_foot") for s in ("left", "right")]
    runner.reset_state(np.zeros(8, dtype=np.float32), phase=0.0)
    runner.latch_heading(sim.read().yaw)
    steps = int(round(seconds * runner.cfg.sim.control_hz))
    w0 = int(round(warmup_s * runner.cfg.sim.control_hz))
    ser = {k: [] for k in ("t", "q", "tau", "roll", "pitch", "vx", "com_z", "foot_z", "foot_vxy_contact", "contact")}
    p_start = None
    for k in range(steps):
        r = sim.read()
        goal = runner.step(r, cmd)
        # physics with per-substep torque logging (same as MujocoTrex.write)
        sim.queue.append(np.asarray(goal, dtype=np.float64).copy())
        target, start = sim.queue[0], sim.goal
        h = m.opt.timestep
        from walk_rl.actuator import lx16a_torque, ramp_reference
        p, v = sim.lx, sim.lx.v_nominal / sim.lx.v_ref
        tau_sub = []
        for i in range(sim.decimation):
            q_ref = ramp_reference(start, target, (i + 1) * h, sim.control_period, p.latency, np.clip)
            d.ctrl[sim.ctrl_idx] = lx16a_torque(q_ref, d.qpos[sim.qadr], d.qvel[sim.dofadr], p.kp, p.deadband, p.coulomb,
                                                p.stall * v, p.no_load * v, p.armature, h, np)
            tau_sub.append(d.ctrl[sim.ctrl_idx].copy())
            mujoco.mj_step(m, d)

        sim.goal = target.copy()
        sim.time += sim.decimation * h
        if k == w0:
            p_start = sim.base_pos()
        if k >= w0:
            R = sim._rot()
            ro, pi = roll_pitch(R)
            yaw = sim.yaw()
            vw = sim.base_vel_world()
            ser["t"].append(sim.time)
            ser["q"].append(d.qpos[sim.qadr].copy())
            ser["tau"].append(np.asarray(tau_sub))  # (decimation, 8)
            ser.setdefault("qd", []).append(d.qvel[sim.dofadr].copy())
            ser["roll"].append(ro)
            ser["pitch"].append(pi)
            ser["vx"].append(math.cos(yaw) * vw[0] + math.sin(yaw) * vw[1])
            ser["com_z"].append(d.subtree_com[sim.base][2])
            ser["foot_z"].append([d.xpos[f][2] for f in feet])
            # foot contact (normal force > 1 N) and the foot's horizontal speed while in contact
            cf = np.zeros(2)
            for ci in range(d.ncon):
                c = d.contact[ci]
                for fi, f in enumerate(feet):
                    if m.geom_bodyid[c.geom1] == f or m.geom_bodyid[c.geom2] == f:
                        fr = np.zeros(6)
                        mujoco.mj_contactForce(m, d, ci, fr)
                        cf[fi] += abs(fr[0])
            vel = np.zeros(6)
            vs = []
            for fi, f in enumerate(feet):
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, f, vel, 0)
                vs.append(math.hypot(vel[3], vel[4]) if cf[fi] > 1.0 else np.nan)
            ser["foot_vxy_contact"].append(vs)
            ser["contact"].append(cf > 1.0)
        if sim.failed():
            break
    out = {"variant": variant or {}, "cmd": cmd, "friction": friction, "delay": delay, "seconds": seconds,
           "survived_s": round(sim.time, 2), "fell": sim.failed()}
    if len(ser["t"]) < 10:
        return out
    q = np.array(ser["q"])
    tau = np.concatenate(ser["tau"], axis=0)  # all physics substeps after warm-up
    frac = np.abs(tau) / stall_v
    # torque spent accelerating the servo rotor (armature), averaged over each control tick, vs the
    # servo torque averaged over the same tick (per-substep accelerations are dominated by contact chatter)
    qd = np.array(ser["qd"])
    arm = np.abs(np.diff(qd, axis=0) / sim.control_period * p.armature) / stall_v
    tick_tau = np.abs(np.array([t.mean(0) for t in ser["tau"]]))[1:] / stall_v
    roll = np.degrees(np.array(ser["roll"]))
    pitch = np.degrees(np.array(ser["pitch"]))
    win = int(2.5 * runner.cfg.sim.control_hz)
    p2p = [np.ptp(roll[i:i + win]) for i in range(0, len(roll) - win + 1, win)] or [np.ptp(roll)]
    dpos = sim.base_pos() - p_start if p_start is not None else np.zeros(3)
    contact = np.array(ser["contact"])
    slip = np.array(ser["foot_vxy_contact"], dtype=float)
    out.update({
        "vx": float(np.mean(ser["vx"])),
        "roll_p2p_deg": float(np.median(p2p)),
        "roll_mean_deg": float(roll.mean()),
        "pitch_mean_deg": float(pitch.mean()),
        "pitch_p2p_deg": float(np.median([np.ptp(pitch[i:i + win]) for i in range(0, len(pitch) - win + 1, win)] or [np.ptp(pitch)])),
        "com_z_mean_m": float(np.mean(ser["com_z"])),
        "side_m": float(dpos[1]),
        "double_support_frac": float(np.mean(contact.all(1))),
        "flight_frac": float(np.mean(~contact.any(1))),
        "foot_slip_mps": float(np.nanmean(slip)) if np.isfinite(slip).any() else None,
        "q_mean_deg": dict(zip(sim.act_names, np.degrees(q.mean(0)).round(2).tolist())),
        "q_p5_p95_deg": dict(zip(sim.act_names, np.degrees(np.percentile(q, [5, 95], axis=0)).T.round(1).tolist())),
        "at_limit_frac": dict(zip(sim.act_names, (np.abs(q) > lim - math.radians(1.0)).mean(0).round(3).tolist())),
        "tau_mean_frac": dict(zip(sim.act_names, frac.mean(0).round(3).tolist())),
        "tau_p95_frac": dict(zip(sim.act_names, np.percentile(frac, 95, axis=0).round(3).tolist())),
        "tau_over80_frac": dict(zip(sim.act_names, (frac > 0.8).mean(0).round(3).tolist())),
    })
    for g, js in LEG_GROUPS.items():
        idx = [sim.act_names.index(j) for j in js]
        out[f"{g}_over80"] = float((frac[:, idx] > 0.8).mean())
        out[f"{g}_mean_frac"] = float(frac[:, idx].mean())
        out[f"{g}_arm_mean_frac"] = float(arm[:, idx].mean())  # |armature * mean qacc over a tick| / stall
        out[f"{g}_tick_tau_mean_frac"] = float(tick_tau[:, idx].mean())
    if keep_series:
        out["series"] = {"q": q.tolist(), "roll": roll.tolist(), "pitch": pitch.tolist(), "foot_z": ser["foot_z"], "contact": contact.tolist()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v9c")
    ap.add_argument("--cmd", type=float, nargs="+", default=[0.0, 0.15, 0.2])
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--friction", type=float, default=0.5)
    ap.add_argument("--variant", default="{}")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run_dir = ROOT / "outputs" / "walk_runs" / a.run
    rows = [run_episode(run_dir, json.loads(a.variant), c, a.seconds, a.friction) for c in a.cmd]
    for r in rows:
        print(json.dumps({k: v for k, v in r.items() if k != "series"}, indent=None))
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
