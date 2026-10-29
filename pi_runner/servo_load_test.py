"""Servo load test: the robot stands on the floor and the servos carry its weight (stage B4).

    python -m pi_runner.servo_load_test              # real robot, on the floor
    python -m pi_runner.servo_load_test --sim        # the same goals in MuJoCo with the lx16a model (PC)

The servo model (robot_config.yaml actuator_model) was fitted to unloaded moves in the air, which do
not identify the stall torque. Here both legs move together (hips, head and tail at zero) so the
robot stays upright, and the leg servos hold and move the body:
  1. zero pose, 2 s
  2. three squat levels (femur/tibia deg below), each a 1 s ramp and a 2 s hold, then back up
  3. squat sine around the middle level, +-6 deg on femur and tibia, 4 s at 1 Hz then 4 s at 2 Hz
  4. zero pose, 2 s
Goals go out at 25 Hz with the policy's move time (one control period), as in walk_live.py.
The static error at each hold gives the load torque / (kp * stall); the sine gives the loaded lag.
Running the same goals in MuJoCo (--sim) gives the model's numbers for the same poses.

Torque goes off at the end, on any exception, on Ctrl-C or ssh drop, on missed servo replies
(walk_live rules), and when roll or pitch moves more than --tilt-deg from the start attitude.
Log: ~/trex/logs/servo_load_<time>.npz (real) or the --out path (sim); same row layout as walk_live.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.robot_config import RobotConfig  # noqa: E402
from pi_runner.runner import ROBOT_CONFIG  # noqa: E402

DT = 0.04
SQUAT_DEG = [(-10.0, -8.0), (-20.0, -16.0), (-28.0, -24.0)]  # (femur, tibia); v9m stance ~(-20, -18)
SINE_DEG = 6.0
MIN_VIN_V = 6.5
MAX_MISSED_READS = 2


def goal_sequence(names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Goals (ticks x joints, rad) and a segment label per tick."""
    leg = {n: k for k, n in enumerate(names)}
    fem = [leg["left_femur_joint"], leg["right_femur_joint"]]
    tib = [leg["left_tibia_joint"], leg["right_tibia_joint"]]

    def pose(f, t):
        g = np.zeros(len(names))
        g[fem], g[tib] = math.radians(f), math.radians(t)
        return g

    out, lab = [], []

    def hold(g, s, name):
        for _ in range(int(round(s / DT))):
            out.append(g.copy()); lab.append(name)

    def ramp(g0, g1, s, name):
        n = int(round(s / DT))
        for k in range(1, n + 1):
            out.append(g0 + (g1 - g0) * k / n); lab.append(name)

    z = pose(0, 0)
    hold(z, 2.0, "zero")
    prev = z
    for i, (f, t) in enumerate(SQUAT_DEG):
        g = pose(f, t)
        ramp(prev, g, 1.0, f"ramp{i + 1}")
        hold(g, 2.0, f"hold{i + 1}")
        prev = g
    for i in reversed(range(len(SQUAT_DEG) - 1)):
        g = pose(*SQUAT_DEG[i])
        ramp(prev, g, 1.0, f"ramp_up{i + 1}")
        hold(g, 2.0, f"hold_up{i + 1}")
        prev = g
    c = pose(*SQUAT_DEG[1])
    for hz in (1.0, 2.0):
        n = int(round(4.0 / DT))
        for k in range(1, n + 1):
            s = math.sin(2 * math.pi * hz * k * DT)
            out.append(c + pose(SINE_DEG * s, SINE_DEG * s)); lab.append(f"sine{hz:.0f}hz")
    ramp(c, z, 1.0, "ramp_zero")
    hold(z, 2.0, "zero_end")
    return np.array(out), lab


def attitude_deg(g) -> tuple[float, float]:
    return math.degrees(math.atan2(g[1], -g[2])), math.degrees(math.atan2(-g[0], -g[2]))


def run(backend, names, goals, labels, tilt_deg, realtime: bool, missed=None):
    missed = missed or (lambda: np.zeros(len(names)))
    rows = []
    rd = backend.read()
    roll0, pitch0 = attitude_deg(rd.gravity_b)
    t0 = time.perf_counter()
    t_next = t0
    for k, goal in enumerate(goals):
        tk = time.perf_counter()
        rd = backend.read()
        roll, pitch = attitude_deg(rd.gravity_b)
        if max(abs(roll - roll0), abs(pitch - pitch0)) > tilt_deg:
            raise RuntimeError(f"tilt roll {roll:+.0f} pitch {pitch:+.0f} deg (start {roll0:+.0f} / {pitch0:+.0f})")
        backend.write(goal, DT) if realtime else backend.write(goal)
        rows.append([k * DT, *rd.q, *goal, *rd.gravity_b, *rd.gyro_b, time.perf_counter() - tk, *missed()])
        if realtime:
            if k % 25 == 0:
                print(f"t {k * DT:5.1f} s {labels[k]:10s} roll {roll:+5.1f} pitch {pitch:+5.1f} | err deg "
                      + " ".join(f"{math.degrees(e):+5.1f}" for e in rd.q - goals[max(k - 1, 0)]), flush=True)
            t_next += DT
            time.sleep(max(0.0, t_next - time.perf_counter()))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="MuJoCo with the lx16a servo model instead of the robot")
    ap.add_argument("--friction", type=float, default=0.5)
    ap.add_argument("--stall-scale", type=float, default=1.0, help="sim only: x stall torque")
    ap.add_argument("--kp-scale", type=float, default=1.0, help="sim only: x firmware P gain")
    ap.add_argument("--tilt-deg", type=float, default=15.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rc = RobotConfig.load(ROBOT_CONFIG)
    names = list(rc.actuated_joints)
    goals, labels = goal_sequence(names)
    meta = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "sim": a.sim, "joints": names, "dt": DT,
            "squat_deg": SQUAT_DEG, "sine_deg": SINE_DEG, "end": None}
    cols = (["t"] + [f"q_{n}" for n in names] + [f"goal_{n}" for n in names]
            + ["gx", "gy", "gz", "wx", "wy", "wz", "work_s"] + [f"missed_{n}" for n in names])
    rows = []
    if a.sim:
        from pi_runner.mujoco_backend import MujocoTrex

        b = MujocoTrex(friction=a.friction, actuator="lx16a")
        b.lx.stall = b.lx.stall * a.stall_scale
        b.lx.kp = b.lx.kp * a.kp_scale
        meta.update(friction=a.friction, stall_scale=a.stall_scale, kp_scale=a.kp_scale)
        for _ in range(25):  # settle on the floor at the zero pose
            b.write(np.zeros(len(names)))
        try:
            rows = run(b, names, goals, labels, a.tilt_deg, realtime=False)
            meta["end"] = "schedule done"
        except RuntimeError as e:
            meta["end"] = f"aborted: {e}"
        out = Path(a.out or f"servo_load_sim_s{a.stall_scale:g}_k{a.kp_scale:g}.npz")
    else:
        from pi_runner.hardware import HardwareBackend

        def _stop(signum, _frame):
            raise KeyboardInterrupt(signal.Signals(signum).name)

        signal.signal(signal.SIGHUP, _stop)
        signal.signal(signal.SIGTERM, _stop)
        hw = HardwareBackend(names, rc.actuator_spec["baud_rate"], [rc.joint_range_rad(n) for n in names],
                             servos=True, max_missed_reads=MAX_MISSED_READS)
        bus = hw.bus
        try:
            vin = [bus.read_vin_v(i) for i in hw.ids]
            meta.update(vin_before=vin, temp_before=[bus.read_temp_c(i) for i in hw.ids])
            print(f"battery {vin} V")
            if any(v is None or v < MIN_VIN_V for v in vin):
                raise RuntimeError(f"servo voltage below {MIN_VIN_V} V: battery off?")
            hw.write(np.zeros(len(names)), 2.0)
            print(f"ramping to the zero pose over 2 s; then {len(goals)} ticks ({len(goals) * DT:.0f} s)", flush=True)
            time.sleep(3.0)
            rows = run(hw, names, goals, labels, a.tilt_deg, realtime=True, missed=lambda: hw.missed.copy())
            meta["end"] = "schedule done"
        except (Exception, KeyboardInterrupt) as e:
            meta["end"] = f"aborted: {e!r}"
            print("ABORT:", e)
        finally:
            for _ in range(2):
                for i in hw.ids:
                    bus.set_torque(i, False)
                time.sleep(0.03)
            print("torque off on all servos")
            try:
                meta.update(vin_after=[bus.read_vin_v(i) for i in hw.ids], temp_after=[bus.read_temp_c(i) for i in hw.ids])
            except Exception as e:  # the log matters more than these numbers
                meta["after_error"] = repr(e)
            hw.imu.close()
            bus.close()
        logs = Path.home() / "trex" / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        out = Path(a.out) if a.out else logs / f"servo_load_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    np.savez(out, rows=np.array(rows, dtype=np.float64), cols=np.array(cols), labels=np.array(labels[:len(rows)]),
             meta=json.dumps(meta))
    print(f"end: {meta['end']}\nlog: {out} ({len(rows)} ticks)")


if __name__ == "__main__":
    main()
