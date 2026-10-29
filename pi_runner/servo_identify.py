"""Servo identification and response logging, robot held in the air. Moves ONE servo at a time.

    python -m pi_runner.servo_identify --check         # read-only: settings, voltage, temperature
    python -m pi_runner.servo_identify --go            # the movement sequence below

Per servo id (ascending), from its current position p0:
  1. ramp: to p0 + 10 deg in 500 ms, hold, back to p0 in 500 ms, hold
  2. step: to p0 + 10 deg with move time 0 (full speed), hold, back, hold
  3. torque off, 1.5 s pause, next servo
The first motion is +10 deg in servo ticks (tick count increasing), or -10 deg where +10 would
leave the servo's stored angle limits; a servo resting outside them is clamped to the limit. During each
move only that servo is polled (~275 Hz), giving its position response for the actuator model.

Aborts, with torque off on every servo, if: any servo is below min_vin_v (battery off) or in
motor mode, a read fails, a servo does not reach its target within tolerance, a temperature
exceeds max_temp_c, or Ctrl-C. Torque is off on every servo at the end in all cases.
Log: ~/trex/logs/servo_identify_<time>.json
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.robot_config import RobotConfig  # noqa: E402
from pi_runner.hardware import load_hardware_config  # noqa: E402
from pi_runner.lx16a import TICKS_PER_DEG, LX16ABus  # noqa: E402
from pi_runner.runner import ROBOT_CONFIG  # noqa: E402

MIN_VIN_V = 6.0
MAX_TEMP_C = 60
DELTA_TICKS = round(10 * TICKS_PER_DEG)  # 42 ticks = 10.08 deg
TOL_TICKS = 8  # ~2 deg


class Abort(Exception):
    pass


def snapshot(bus, ids):
    out = {}
    for i in ids:
        out[i] = {"pos": bus.read_pos(i), "vin_v": bus.read_vin_v(i), "temp_c": bus.read_temp_c(i)}
    return out


def track(bus, sid, target, dur_s, t0, log):
    end = time.perf_counter() + dur_s
    p = None
    while time.perf_counter() < end:
        p = bus.read_pos(sid)
        if p is None:
            raise Abort(f"id {sid}: no position reply while moving")
        log.append([round(time.perf_counter() - t0, 5), target, p])
    return p


def move_and_track(bus, sid, target, move_ms, hold_s, t0, log):
    bus.move(sid, target, move_ms)
    log.append([round(time.perf_counter() - t0, 5), target, None])  # command instant
    p = track(bus, sid, target, move_ms / 1000 + hold_s, t0, log)
    if abs(p - target) > TOL_TICKS:
        raise Abort(f"id {sid}: at {p} ticks, target {target} (blocked or too weak?)")
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--go", action="store_true")
    ap.add_argument("--ids", type=int, nargs="*", default=list(range(1, 9)))
    a = ap.parse_args()

    def _stop(signum, _frame):  # ssh drop (SIGHUP) or kill: same path as Ctrl-C, torque off
        raise KeyboardInterrupt(signal.Signals(signum).name)

    signal.signal(signal.SIGHUP, _stop)
    signal.signal(signal.SIGTERM, _stop)

    hw = load_hardware_config()
    bus = LX16ABus(hw["servo_bus"]["port"], RobotConfig.load(ROBOT_CONFIG).actuator_spec["baud_rate"],
                   hw["servo_bus"]["read_timeout_s"])
    ids = a.ids
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "mode": "go" if a.go else "check", "ids": ids,
              "delta_ticks": DELTA_TICKS, "settings": {}, "before": None, "servos": {}, "after": None, "result": None}
    try:
        for i in ids:
            s = bus.read_settings(i)
            if s is None:
                raise Abort(f"id {i}: no reply to settings reads")
            record["settings"][i] = s
        record["before"] = snapshot(bus, ids)
        for i in ids:
            s, b = record["settings"][i], record["before"][i]
            print(f"id {i}: pos {b['pos']:4d}  {b['vin_v']:.2f} V  {b['temp_c']} C  offset {s['angle_offset_ticks']:+d}  "
                  f"limits {s['angle_limit_ticks']}  vin limits {s['vin_limit_v']}  "
                  f"{'MOTOR MODE' if s['motor_mode'] else 'servo mode'}  torque {'on' if s['torque_on'] else 'off'}")
        low = [i for i in ids if record["before"][i]["vin_v"] < MIN_VIN_V]
        if low:
            raise Abort(f"servo voltage below {MIN_VIN_V} V on ids {low}: battery off?")
        motor = [i for i in ids if record["settings"][i]["motor_mode"]]
        if motor:
            raise Abort(f"ids {motor} are in motor (continuous rotation) mode")
        if a.check:
            record["result"] = "check ok"
            print("check ok: ready to move")
            return
        t0 = time.perf_counter()
        for i in ids:
            p0 = record["before"][i]["pos"]
            lo, hi = record["settings"][i]["angle_limit_ticks"]
            lo, hi = max(lo, 0), min(hi, 1000)
            home = min(max(p0, lo), hi)  # a servo resting outside its stored limits is clamped there
            sign = 1 if home + DELTA_TICKS <= hi else -1
            away = home + sign * DELTA_TICKS
            if not lo <= away <= hi:
                raise Abort(f"id {i}: no room for 10 deg inside its angle limits [{lo}, {hi}] from {home}")
            rec = {"p0": p0, "home": home, "away": away, "sign": sign,
                   "start_s": round(time.perf_counter() - t0, 3), "log": []}
            record["servos"][i] = rec
            print(f"[{rec['start_s']:6.1f} s] id {i}: {'+' if sign > 0 else '-'}10 deg (ticks {home} -> {away})"
                  f"{' from ' + str(p0) + ' clamped to its limit' if home != p0 else ''}: ramp, back, step, back",
                  flush=True)
            v0 = bus.read_vin_v(i)
            move_and_track(bus, i, away, 500, 0.5, t0, rec["log"])
            rec["vin_loaded_v"] = bus.read_vin_v(i)
            move_and_track(bus, i, home, 500, 0.5, t0, rec["log"])
            move_and_track(bus, i, away, 0, 0.6, t0, rec["log"])
            move_and_track(bus, i, home, 0, 0.6, t0, rec["log"])
            bus.set_torque(i, False)
            rec["vin_before_v"], rec["temp_after_c"] = v0, bus.read_temp_c(i)
            if rec["temp_after_c"] is not None and rec["temp_after_c"] > MAX_TEMP_C:
                raise Abort(f"id {i}: {rec['temp_after_c']} C")
            time.sleep(1.5)
        record["after"] = snapshot(bus, ids)
        record["result"] = "done"
        print("sequence done")
    except (Abort, KeyboardInterrupt) as e:
        record["result"] = f"aborted: {e!r}"
        print("ABORT:", e)
    finally:
        for i in ids:
            bus.set_torque(i, False)
        time.sleep(0.05)
        for i in ids:
            bus.set_torque(i, False)  # twice: a write can be lost on the bus
        print("torque off on all servos")
        logs = Path.home() / "trex" / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        f = logs / f"servo_identify_{time.strftime('%Y%m%d_%H%M%S')}_{record['mode']}.json"
        f.write_text(json.dumps(record))
        print("log:", f)
        bus.close()


if __name__ == "__main__":
    main()
