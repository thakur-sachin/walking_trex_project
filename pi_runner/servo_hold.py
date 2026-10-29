"""Stand stage 1: drive all 8 joints to the zero pose and hold it, robot held in the air.

    python -m pi_runner.servo_hold --check    # read-only: voltages, current joint angles
    python -m pi_runner.servo_hold --go       # ramp to zero in 2 s, hold 10 s, torque off
    python -m pi_runner.servo_hold --go --hold-s 90   # longer hold, e.g. the tilt (friction) test on a plank
    python -m pi_runner.servo_hold --sweep    # stage 1b: each joint alone to +10 deg and back

Goes through HardwareBackend (the calibrated deployment path); the servos' stored limits
clamp every command as well. A servo may miss its position reply on MAX_MISSED_READS ticks in a row
(the last good position is used). Torque off on every servo at the end, on any exception, if a
joint is more than MAX_ERR_DEG from zero after the ramp, or on Ctrl-C / ssh drop.
While holding it prints the IMU tilt (pitch, roll) every second; the log keeps it per tick, so a tilt test
gets the plank angle from the IMU as well as from the inclinometer.
Log: ~/trex/logs/servo_hold_<time>.json
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
from pi_runner.hardware import HardwareBackend  # noqa: E402
from pi_runner.runner import ROBOT_CONFIG  # noqa: E402

MIN_VIN_V = 6.5
RAMP_S = 2.0
HOLD_S = 10.0
MAX_ERR_DEG = 5.0
MAX_MISSED_READS = 2  # consecutive ticks a servo may miss its position reply (as walk_live.py)


SWEEP_DEG = 10.0


def track(hw, dur_s, t0, log):
    end = time.perf_counter() + dur_s
    q = None
    while time.perf_counter() < end:
        tk = time.perf_counter()
        q = hw.read().q
        log.append([round(tk - t0, 4)] + np.degrees(q).round(2).tolist())
        time.sleep(max(0.0, 0.04 - (time.perf_counter() - tk)))
    return q


def sweep(hw, names, rec):
    """Each joint alone: 0 -> +SWEEP_DEG in 1 s, hold 1 s, back to 0 in 1 s, 1.5 s pause."""
    t0 = time.perf_counter()
    hw.write(np.zeros(len(names)), 1.0)
    track(hw, 1.5, t0, rec["log"])
    rec["sweep"] = {}
    for k, n in enumerate(names):
        goal = np.zeros(len(names))
        goal[k] = math.radians(SWEEP_DEG)
        print(f"[{time.perf_counter() - t0:5.1f} s] {n}: +{SWEEP_DEG:.0f} deg", flush=True)
        hw.write(goal, 1.0)
        q = track(hw, 2.0, t0, rec["log"])
        reached = math.degrees(q[k])
        others = np.delete(np.degrees(q), k)
        rec["sweep"][n] = {"reached_deg": round(reached, 2), "others_max_abs_deg": round(float(np.abs(others).max()), 2)}
        print(f"           reached {reached:+.1f} deg (others within {np.abs(others).max():.1f})", flush=True)
        if abs(reached - SWEEP_DEG) > MAX_ERR_DEG:
            raise RuntimeError(f"{n} reached {reached:+.1f} deg, target +{SWEEP_DEG:.0f}")
        hw.write(np.zeros(len(names)), 1.0)
        track(hw, 2.5, t0, rec["log"])
    rec["result"] = "sweep done"
    print("sweep done")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--go", action="store_true")
    g.add_argument("--sweep", action="store_true")
    ap.add_argument("--hold-s", type=float, default=HOLD_S, help="hold time for --go, s")
    a = ap.parse_args()

    def _stop(signum, _frame):
        raise KeyboardInterrupt(signal.Signals(signum).name)

    signal.signal(signal.SIGHUP, _stop)
    signal.signal(signal.SIGTERM, _stop)

    rc = RobotConfig.load(ROBOT_CONFIG)
    names = list(rc.actuated_joints)
    hw = HardwareBackend(names, rc.actuator_spec["baud_rate"], [rc.joint_range_rad(n) for n in names], servos=True,
                         max_missed_reads=MAX_MISSED_READS)
    bus = hw.bus
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "joints": names, "ids": hw.ids, "log": [], "result": None}
    try:
        vin = [bus.read_vin_v(i) for i in hw.ids]
        temp = [bus.read_temp_c(i) for i in hw.ids]
        q0 = hw.read().q
        rec.update(vin_before=vin, temp_before=temp, q_before_deg=np.degrees(q0).round(2).tolist())
        for n, v, t, q in zip(names, vin, temp, q0):
            print(f"{n:18s} {v} V  {t} C  now {math.degrees(q):+6.1f} deg")
        if any(v is None or v < MIN_VIN_V for v in vin):
            raise RuntimeError(f"servo voltage below {MIN_VIN_V} V: battery off?")
        if a.check:
            rec["result"] = "check ok"
            print("check ok")
            return
        zero = np.zeros(len(names))
        if a.sweep:
            sweep(hw, names, rec)
            return
        t0 = time.perf_counter()
        hw.write(zero, RAMP_S)
        print(f"ramping to the zero pose over {RAMP_S:.0f} s, then holding {a.hold_s:.0f} s", flush=True)
        checked, next_print = False, RAMP_S
        rec["tilt_log"] = []  # t, pitch, roll [deg] from the IMU gravity vector
        while time.perf_counter() - t0 < RAMP_S + a.hold_s:
            tk = time.perf_counter()
            r = hw.read()
            q = r.q
            rec["log"].append([round(tk - t0, 4)] + np.degrees(q).round(2).tolist())
            g = np.asarray(r.gravity_b, dtype=float) / np.linalg.norm(r.gravity_b)
            pitch, roll = math.degrees(math.asin(max(-1.0, min(1.0, g[0])))), math.degrees(math.atan2(-g[1], -g[2]))
            rec["tilt_log"].append([round(tk - t0, 4), round(pitch, 2), round(roll, 2)])
            if tk - t0 >= next_print:
                next_print += 1.0
                print(f"t {tk - t0:5.1f} s  pitch {pitch:+5.1f}  roll {roll:+5.1f} deg", flush=True)
            if not checked and tk - t0 > RAMP_S + 0.5:
                checked = True
                err = np.degrees(np.abs(q))
                print("after ramp [deg]: " + " ".join(f"{n.replace('_joint', '')} {e:+.1f}"
                                                        for n, e in zip(names, np.degrees(q))), flush=True)
                if err.max() > MAX_ERR_DEG:
                    raise RuntimeError(f"{names[int(err.argmax())]} is {err.max():.1f} deg from zero")
            time.sleep(max(0.0, 0.04 - (time.perf_counter() - tk)))
        L = np.array([r[1:] for r in rec["log"] if r[0] > RAMP_S + 0.5])
        rec["hold_mean_deg"] = L.mean(0).round(2).tolist()
        rec["hold_p2p_deg"] = (L.max(0) - L.min(0)).round(2).tolist()
        rec["vin_holding"] = [bus.read_vin_v(i) for i in hw.ids]
        rec["temp_after"] = [bus.read_temp_c(i) for i in hw.ids]
        print("hold mean / peak-to-peak [deg]:")
        for n, m, p in zip(names, rec["hold_mean_deg"], rec["hold_p2p_deg"]):
            print(f"  {n:18s} {m:+5.1f}  {p:4.1f}")
        print(f"battery while holding {rec['vin_holding']}; temps {rec['temp_after']}")
        rec["result"] = "done"
    except (Exception, KeyboardInterrupt) as e:
        rec["result"] = f"aborted: {e!r}"
        print("ABORT:", e)
    finally:
        for _ in range(2):
            for i in hw.ids:
                bus.set_torque(i, False)
            time.sleep(0.03)
        print("torque off on all servos")
        logs = Path.home() / "trex" / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        f = logs / f"servo_hold_{time.strftime('%Y%m%d_%H%M%S')}{'_check' if a.check else '_sweep' if a.sweep else ''}.json"
        f.write_text(json.dumps(rec))
        print("log:", f)
        hw.imu.close()
        bus.close()


if __name__ == "__main__":
    main()
