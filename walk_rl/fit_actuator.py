"""Fit the LX-16A servo model (walk_rl/actuator.py) to the logged real-servo moves.

    python walk_rl/fit_actuator.py [--dt 0.005] [--workers 8]

Data (assets/robots/trex/servo_logs/, logged on the Pi 2026-10-18, robot held in the air):
  servo_identify_20261018.json  pi_runner/servo_identify.py --go: each servo alone, 10 deg ramp
                                (0.5 s) and full-speed step out and back, position every ~3.75 ms
  servo_sweep_20261018.json     pi_runner/servo_hold.py --sweep: every servo holding, one joint at a
                                time to +10 deg in 1 s and back; only the settled part is scored
Each segment is simulated as one joint with the exact actuator code used by the simulators
(semi-implicit Euler at the physics step), the link inertia and gravity taken from the MuJoCo
model at the segment's mid angle (mass matrix and qfrc_bias), and the fitted parameters:
kp, deadband, latency, armature, Coulomb friction, no-load speed scale (stall torque fixed at the
datasheet value, both scaled by the measured battery voltage). Minimised: RMS position error over
all segments (scipy differential_evolution). Result: outputs/actuator_fit.json; the nominal values
go into robot_config.yaml actuator_model by hand.

The identify log predates the final calibration, so its ticks are converted with the zero ticks
in force at the time (ZERO_AT_LOG; the tail horn was later re-indexed).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.robot_config import RobotConfig  # noqa: E402
from pi_runner.mujoco_backend import build_model  # noqa: E402
from pi_runner.runner import ROBOT_CONFIG  # noqa: E402
from walk_rl.actuator import lx16a_torque, ramp_reference  # noqa: E402

LOGS = PROJECT_ROOT / "assets" / "robots" / "trex" / "servo_logs"
ID2J = {1: "left_hip_joint", 2: "left_femur_joint", 3: "left_tibia_joint", 4: "right_hip_joint",
        5: "right_femur_joint", 6: "right_tibia_joint", 7: "head_joint", 8: "tail_joint"}
DIR = {"head_joint": 1, "tail_joint": 1, "left_hip_joint": -1, "right_hip_joint": 1, "left_femur_joint": 1,
       "right_femur_joint": 1, "left_tibia_joint": 1, "right_tibia_joint": 1}
ZERO_AT_LOG = {"head_joint": 448, "tail_joint": -49, "left_hip_joint": 327, "right_hip_joint": 396,
               "left_femur_joint": 168, "right_femur_joint": 408, "left_tibia_joint": 387, "right_tibia_joint": 343}
SWEEP_CMD_T = [1.5, 6.1, 10.6, 15.1, 19.6, 24.2, 28.7, 33.2]  # command instants in the sweep log (+-0.05 s)
T2R = math.radians(0.24)
V_REF = 7.4
BOUNDS = [(0.5, 30), (0.0, math.radians(3)), (0.0, 0.06), (0.0005, 0.06), (0.0, 0.6), (0.2, 1.5)]
NAMES = ["kp", "deadband_rad", "latency_s", "armature_kgm2", "coulomb_nm", "no_load_speed_scale"]

rc = RobotConfig.load(ROBOT_CONFIG)
model = build_model(rc)
data = mujoco.MjData(model)


def inertia_and_gravity(joint: str, theta: float) -> tuple[float, float]:
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    dof, qa = model.jnt_dofadr[j], model.jnt_qposadr[j]
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.5
    data.qpos[qa] = theta
    mujoco.mj_forward(model, data)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    return M[dof, dof], -data.qfrc_bias[dof]


def segment(joint, kind, move_s, t, p, p0, tgt, mask=None):
    mid = 0.5 * (p0 + tgt)
    J, g0 = inertia_and_gravity(joint, mid)
    _, g1 = inertia_and_gravity(joint, mid + 0.05)
    return dict(j=joint, kind=kind, move=move_s, t=t, p=p, p0=p0, tgt=tgt, J=J, g0=g0, gk=(g1 - g0) / 0.05, mid=mid,
                mask=np.ones(len(t), bool) if mask is None else mask)


def load_segments() -> tuple[list[dict], float]:
    r = json.loads((LOGS / "servo_identify_20261018.json").read_text())
    volts = float(np.mean([r["before"][k]["vin_v"] for k in r["before"]]))
    segs = []
    for sid, s in r["servos"].items():
        jn = ID2J[int(sid)]
        rad = lambda tick: DIR[jn] * (tick - ZERO_AT_LOG[jn]) * T2R  # noqa: E731
        log = s["log"]
        cmds = [k for k, x in enumerate(log) if x[2] is None]  # [t, target, None] marks a command
        for n, ci in enumerate(cmds):
            tc, tgt, _ = log[ci]
            seg = log[ci + 1:cmds[n + 1] if n + 1 < len(cmds) else len(log)]
            t = np.array([x[0] - tc for x in seg])
            p = np.array([rad(x[2]) for x in seg])
            p0 = rad(log[ci - 1][2]) if ci > 0 and log[ci - 1][2] is not None else p[0]
            segs.append(segment(jn, ["ramp+", "ramp-", "step+", "step-"][n], [0.5, 0.5, 0, 0][n], t, p, p0, rad(tgt)))
    sw = json.loads((LOGS / "servo_sweep_20261018.json").read_text())
    L = np.array(sw["log"])
    for k, jn in enumerate(sw["joints"]):
        for up in (True, False):
            tc = SWEEP_CMD_T[k] + (0.0 if up else 2.0)
            m = (L[:, 0] >= tc) & (L[:, 0] < tc + (2.0 if up else 2.5))
            t, p = L[m, 0] - tc, np.radians(L[m, 1 + k])
            segs.append(segment(jn, "hold+" if up else "hold0", 1.0, t, p, p[0], math.radians(10.0) if up else 0.0,
                                mask=t > 1.3))
    return segs, volts


def simulate(seg, x, volts, dt):
    kp, db, lat, arm, fc, wscale = x
    stall = rc.actuator_stall_torque_nm * volts / V_REF
    no_load = rc.actuator_no_load_speed_rad_per_sec * wscale * volts / V_REF
    J, n = seg["J"] + arm, int(seg["t"][-1] / dt) + 2
    th, w, out = seg["p0"], 0.0, np.empty(n)
    period = max(seg["move"], 1e-9)
    for i in range(n):
        ref = ramp_reference(seg["p0"], seg["tgt"], (i + 1) * dt, period, lat, np.clip)
        tau = lx16a_torque(ref, th, w, kp, db, fc, stall, no_load, arm, dt, np) + seg["g0"] + seg["gk"] * (th - seg["mid"])
        w += tau / J * dt
        th += w * dt
        out[i] = th
    return np.interp(seg["t"], (np.arange(n) + 1) * dt, out)


def rms_deg(x, segs, volts, dt):
    e = np.concatenate([(simulate(s, x, volts, dt) - s["p"])[s["mask"]] for s in segs])
    return math.degrees(math.sqrt(np.mean(e ** 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dt", type=float, default=0.005, help="physics step of the simulators (200 Hz)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "actuator_fit.json"))
    a = ap.parse_args()
    segs, volts = load_segments()
    print(f"{len(segs)} segments, battery {volts:.2f} V")
    res = differential_evolution(rms_deg, BOUNDS, args=(segs, volts, a.dt), seed=a.seed, maxiter=40, popsize=12,
                                 tol=1e-6, polish=True, workers=a.workers, updating="deferred")
    x = res.x
    print(f"fit RMS {res.fun:.3f} deg | kp {x[0]:.2f} duty/rad (saturates {math.degrees(1 / x[0]):.1f} deg from the goal) "
          f"deadband {math.degrees(x[1]):.2f} deg, latency {x[2] * 1e3:.1f} ms, armature {x[3]:.4f} kg m2, "
          f"Coulomb {x[4]:.3f} N m, no-load speed x{x[5]:.2f}")
    for kind in ("ramp+", "ramp-", "step+", "step-", "hold+", "hold0"):
        print(f"  {kind} RMS deg: " + " ".join(
            f"{s['j'].replace('_joint', '')}={rms_deg(x, [s], volts, a.dt):.2f}" for s in segs if s["kind"] == kind))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"params": dict(zip(NAMES, map(float, x))), "rms_deg": float(res.fun), "battery_v": volts,
                               "dt": a.dt}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
