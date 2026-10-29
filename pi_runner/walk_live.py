"""Run a policy on the real robot and log every control tick (stages: hanging, then on the floor).

    python -m pi_runner.walk_live RUN_DIR --schedule 0:3 0.10:10 0:3            # robot hanging
    python -m pi_runner.walk_live RUN_DIR --schedule 0:3 0.10:10 0:3 --floor    # robot on the floor

Sequence: battery check (every servo >= 6.5 V), ramp to the zero pose over 2 s, hold 1 s, then the
policy at 25 Hz through PolicyRunner + HardwareBackend (the path pi_runner/test_obs_parity.py checks
against training). `--schedule` is a list of cmd_m_per_s:seconds.

Torque goes off on every servo at the end, on any exception, when a servo misses its position
reply on more than MAX_MISSED_READS ticks in a row (on fewer, its last good position is used and the
miss is logged), on Ctrl-C or
ssh drop, when the runner's tilt safety trips, and (hanging) when roll or pitch moves more than
--hang-tilt-deg from the attitude at the start: a hanging robot that swings that far is being pushed by
its legs into something.
Floor runs use the training tilt limit (termination.max_tilt_deg) via the runner's safety.

Log: ~/trex/logs/walk_<run>_<time>.npz, one row per tick: time, command, gait phase, measured joints,
servo goals, policy action, gravity vector, gyro, yaw, tick work time; plus the observations, the
battery voltage and servo temperatures before and after, and why the run ended. Each tick also
reads the supply voltage of one servo (round robin over the ids, when the tick has time left), to see
voltage dips under load. If the USB serial adapter drops off the bus during the run (it did on
2026-10-24), the torque-off reopens the port and tries again, and the log is saved in every case.
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
from pi_runner.lx16a import LX16ABus  # noqa: E402
from pi_runner.runner import ROBOT_CONFIG, PolicyRunner  # noqa: E402

MIN_VIN_V = 6.5
RAMP_S = 2.0
SETTLE_S = 1.0
MAX_START_ERR_DEG = 5.0
MAX_MISSED_READS = 2  # consecutive ticks a servo may miss its position reply (last good value used)
VIN_MIN_SLACK_S = 0.008  # read one servo's voltage in a tick only if this much of the tick is left
REOPEN_WAIT_S = 5.0  # how long the torque-off waits for a reset USB adapter to come back


def parse_schedule(items: list[str]) -> list[tuple[float, float]]:
    out = []
    for it in items:
        c, s = it.split(":")
        out.append((float(c), float(s)))
    return out


def torque_off(hw, baud: int) -> str:
    """Torque off on every servo, twice. On a serial error (adapter reset) reopen the port and retry."""
    t_end = time.perf_counter() + REOPEN_WAIT_S
    err = None
    while True:
        try:
            for _ in range(2):
                for i in hw.ids:
                    hw.bus.set_torque(i, False)
                time.sleep(0.03)
            return "ok" if err is None else f"ok after reopening the port ({err!r})"
        except Exception as e:  # serial.SerialException / OSError when the adapter drops off
            err = e
            if time.perf_counter() > t_end:
                return f"FAILED: {e!r}"
            try:
                hw.bus.close()
            except Exception:
                pass
            time.sleep(0.5)
            try:
                hw.bus = LX16ABus(hw.bus.port, baud, hw.bus.timeout)
            except Exception as e2:
                err = e2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--schedule", nargs="+", default=["0:3", "0.10:10", "0:3"])
    ap.add_argument("--floor", action="store_true", help="robot on the floor (default: hanging)")
    ap.add_argument("--hang-tilt-deg", type=float, default=20.0)
    ap.add_argument("--read-order", choices=["joints", "reversed"], default="joints",
                    help="order of the servo position reads in each tick (reversed: bus diagnosis)")
    a = ap.parse_args()

    def _stop(signum, _frame):
        raise KeyboardInterrupt(signal.Signals(signum).name)

    signal.signal(signal.SIGHUP, _stop)
    signal.signal(signal.SIGTERM, _stop)

    sched = parse_schedule(a.schedule)
    r = PolicyRunner(a.run)
    rc = RobotConfig.load(ROBOT_CONFIG)
    names = r.act_names
    hw = HardwareBackend(names, rc.actuator_spec["baud_rate"], [rc.joint_range_rad(n) for n in names], servos=True,
                         max_missed_reads=MAX_MISSED_READS)
    hw.set_read_order(a.read_order)
    bus = hw.bus
    run = Path(a.run).name
    meta = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "run": run, "floor": a.floor, "schedule": sched,
            "joints": names, "ids": hw.ids, "read_order": a.read_order, "dt": r.dt, "end": None}
    rows, obs, vin_log = [], [], []  # vin_log: (tick, servo id, volts or nan)
    try:
        vin = [bus.read_vin_v(i) for i in hw.ids]
        meta.update(vin_before=vin, temp_before=[bus.read_temp_c(i) for i in hw.ids])
        print(f"battery {vin} V")
        if any(v is None or v < MIN_VIN_V for v in vin):
            raise RuntimeError(f"servo voltage below {MIN_VIN_V} V: battery off?")
        zero = np.zeros(len(names))
        hw.write(zero, RAMP_S)
        print(f"ramping to the zero pose over {RAMP_S:.0f} s", flush=True)
        time.sleep(RAMP_S + SETTLE_S)
        first = hw.read()
        err = np.degrees(np.abs(first.q))
        if err.max() > MAX_START_ERR_DEG:
            raise RuntimeError(f"{names[int(err.argmax())]} is {err.max():.1f} deg from zero after the ramp")
        r.reset_state(zero, yaw_ref=first.yaw)
        g0 = first.gravity_b
        roll0, pitch0 = math.degrees(math.atan2(g0[1], -g0[2])), math.degrees(math.atan2(-g0[0], -g0[2]))
        meta.update(roll0_deg=roll0, pitch0_deg=pitch0)
        cmds = np.concatenate([np.full(int(round(s / r.dt)), c) for c, s in sched])
        print(f"policy {run}: {len(cmds)} ticks, schedule {sched}, {'floor' if a.floor else 'hanging'}", flush=True)
        t0 = time.perf_counter()
        t_next = t0
        for k, cmd in enumerate(cmds):
            tk = time.perf_counter()
            rd = hw.read()
            phase = float(r.phase)
            goal = r.step(rd, float(cmd))
            if r.safety.tripped:
                raise RuntimeError(f"safety: {r.safety.tripped}")
            g = rd.gravity_b
            roll, pitch = math.degrees(math.atan2(g[1], -g[2])), math.degrees(math.atan2(-g[0], -g[2]))
            if not a.floor and max(abs(roll - roll0), abs(pitch - pitch0)) > a.hang_tilt_deg:
                raise RuntimeError(f"hanging tilt roll {roll:+.0f} pitch {pitch:+.0f} deg (start {roll0:+.0f} / {pitch0:+.0f})")
            hw.write(goal, r.dt)
            work = time.perf_counter() - tk
            if t_next + r.dt - time.perf_counter() > VIN_MIN_SLACK_S:
                sid = hw.ids[k % len(hw.ids)]
                v = bus.read_vin_v(sid)
                vin_log.append((k, sid, np.nan if v is None else v))
            rows.append([tk - t0, cmd, phase, *rd.q, *goal, *r.last_action, *g, *rd.gyro_b, rd.yaw, work, *hw.missed])
            obs.append(r.last_obs.copy())
            if k % int(round(1 / r.dt)) == 0:
                print(f"t {tk - t0:5.1f} s cmd {cmd:.2f} roll {roll:+5.1f} pitch {pitch:+5.1f} | q "
                      + " ".join(f"{math.degrees(v):+5.0f}" for v in rd.q) + f" | work {work * 1000:.0f} ms", flush=True)
            t_next += r.dt
            time.sleep(max(0.0, t_next - time.perf_counter()))
        meta["end"] = "schedule done"
    except (Exception, KeyboardInterrupt) as e:
        meta["end"] = f"aborted: {e!r}"
        print("ABORT:", e)
    finally:
        meta["torque_off"] = torque_off(hw, rc.actuator_spec["baud_rate"])
        bus = hw.bus
        print(f"torque off on all servos: {meta['torque_off']}")
        try:
            meta.update(vin_after=[bus.read_vin_v(i) for i in hw.ids], temp_after=[bus.read_temp_c(i) for i in hw.ids])
        except Exception as e:  # the log matters more than these numbers
            meta["after_error"] = repr(e)
        if rows:
            m = np.array(rows)[:, -len(names):]
            meta["missed_ticks"] = {n: int((m[:, k] > 0).sum()) for k, n in enumerate(names) if (m[:, k] > 0).any()}
            print(f"missed position replies (ticks): {meta['missed_ticks'] or 'none'}")
        cols = (["t", "cmd", "phase"] + [f"q_{n}" for n in names] + [f"goal_{n}" for n in names]
                + [f"action_{n}" for n in names] + ["gx", "gy", "gz", "wx", "wy", "wz", "yaw", "work_s"] + [f"missed_{n}" for n in names])
        logs = Path.home() / "trex" / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        f = logs / f"walk_{run}_{time.strftime('%Y%m%d_%H%M%S')}{'_floor' if a.floor else '_hang'}.npz"
        np.savez(f, rows=np.array(rows, dtype=np.float64), cols=np.array(cols), obs=np.array(obs, dtype=np.float32),
                 vin=np.array(vin_log, dtype=np.float64).reshape(-1, 3), meta=json.dumps(meta))
        print(f"end: {meta['end']}\nlog: {f} ({len(rows)} ticks)")
        if vin_log:
            v = np.array(vin_log)[:, 2]
            print(f"servo supply during the run: min {np.nanmin(v):.2f} V, mean {np.nanmean(v):.2f} V, "
                  f"{int(np.isnan(v).sum())}/{len(v)} reads missed")
        for closer in (hw.imu.close, bus.close):
            try:
                closer()
            except Exception:
                pass


if __name__ == "__main__":
    main()
