"""Bring-up checks on the Pi. Nothing here moves a servo.

    python -m pi_runner.hw_check info                 # board, clocks, throttling, buses, adapters
    python -m pi_runner.hw_check imu --seconds 20     # live BNO055: frames, rate, read latency
    python -m pi_runner.hw_check bench RUN_DIR        # policy step time, goals dump for PC parity
    python -m pi_runner.hw_check dryrun RUN_DIR       # real IMU + policy at 25 Hz, servos untouched
    python -m pi_runner.hw_check bus [--scan]         # BusLinker: ids, position, voltage, temperature
    python -m pi_runner.hw_check watch --seconds 90   # record all servo positions (hand-moved joints)

`bench --dump` on the PC and on the Pi writes the goals for the same synthetic readings;
`compare A B` reports how many servo ticks differ.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pi_runner.runner import ROBOT_CONFIG, PolicyRunner, Reading  # noqa: E402


def _sh(path: str) -> str:
    try:
        return Path(path).read_text().strip().replace("\x00", "")
    except OSError:
        return "-"


def _stats(ms: np.ndarray) -> str:
    return (f"mean {ms.mean():.3f} ms, p50 {np.percentile(ms, 50):.3f}, p99 {np.percentile(ms, 99):.3f}, "
            f"max {ms.max():.3f}")


# --------------------------------------------------------------------- info
def cmd_info(_):
    import subprocess

    def run(*a):
        try:
            return subprocess.run(a, capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "-"

    print("board     ", _sh("/proc/device-tree/model"))
    print("kernel    ", platform.release(), platform.machine())
    print("python    ", sys.version.split()[0], " numpy", np.__version__)
    print("governor  ", _sh("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
          " cur MHz", int(_sh("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") or 0) // 1000)
    print("temp      ", run("vcgencmd", "measure_temp"))
    t = run("vcgencmd", "get_throttled")
    print("throttled ", t, "(0x0 = never under-voltage/throttled since boot)")
    print("i2c       ", " ".join(sorted(glob.glob("/dev/i2c-*"))) or "none (enable I2C)")
    print("uart      ", " ".join(sorted(glob.glob("/dev/serial[0-9]") + glob.glob("/dev/ttyAMA*"))) or "none")
    ports = sorted(glob.glob("/dev/serial/by-id/*"))
    print("usb serial", ports or "none plugged in (servo adapter not connected)")
    for p in ports:
        tty = os.path.basename(os.path.realpath(p))
        lt = _sh(f"/sys/bus/usb-serial/devices/{tty}/latency_timer")
        print(f"   {tty}: latency_timer {lt} ms" if lt != "-" else f"   {tty}: no latency_timer (CH34x/CP210x)")


# ---------------------------------------------------------------------- imu
def cmd_imu(a):
    from pi_runner.bno055 import BNO055
    from pi_runner.hardware import load_hardware_config

    imu = BNO055.from_config(load_hardware_config()["imu"])
    print(f"BNO055 at 0x{imu.addr:02x}, mode {imu.mode}; status/err {imu.status()}")
    print("Check the frame: level -> g ~ (0, 0, -1); nose down -> g_x > 0; left side down -> g_y > 0;\n"
          "turn left -> gyro_z > 0 and yaw increases. If not, fix imu.sensor_to_body in hardware.yaml.\n")
    lat, t_end, last_print, n, last, fresh = [], time.time() + a.seconds, 0.0, 0, None, 0
    while time.time() < t_end:
        t0 = time.perf_counter()
        g, w, yaw = imu.read()
        lat.append((time.perf_counter() - t0) * 1000)
        n += 1
        sample = (tuple(g), tuple(w))
        fresh += sample != last
        last = sample
        if time.time() - last_print > 0.25:
            last_print = time.time()
            c = imu.calibration()
            print(f"g ({g[0]:+.3f} {g[1]:+.3f} {g[2]:+.3f})  gyro ({w[0]:+.3f} {w[1]:+.3f} {w[2]:+.3f}) rad/s  "
                  f"yaw {math.degrees(yaw):+7.2f} deg  calib sys{c['sys']} gyr{c['gyro']} acc{c['accel']}", flush=True)
        time.sleep(max(0.0, 0.01 - (time.perf_counter() - t0)))
    lat = np.array(lat)
    print(f"\n{n} reads in {a.seconds:.0f} s ({n / a.seconds:.0f}/s), {fresh} changed samples "
          f"({fresh / a.seconds:.0f}/s), I2C retries {imu.errors}")
    print("read latency:", _stats(lat))
    # Bit-7 corruption check (hardware I2C + BNO055 clock stretching): static registers
    # must read back exactly what _begin() wrote, every time.
    from pi_runner.bno055 import MODES, OPR_MODE, UNIT_SEL

    bad = sum(imu._r(UNIT_SEL) != 0x06 for _ in range(500)) + sum(imu._r(OPR_MODE) != MODES[imu.mode] for _ in range(500))
    print(f"register integrity: {bad} of 1000 static-register reads wrong" + (" -> I2C is corrupting data" if bad else ""))


# -------------------------------------------------------------------- bench
def synthetic_readings(n: int, seed: int = 0):
    """Deterministic, plausible readings: a slow body wobble and joint motion."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 25.0
    for i in range(n):
        roll, pitch = 0.08 * math.sin(2 * math.pi * 1.1 * t[i]), 0.05 * math.sin(2 * math.pi * 0.7 * t[i])
        g = np.array([math.sin(pitch), -math.sin(roll) * math.cos(pitch), -math.cos(roll) * math.cos(pitch)])
        gyro = rng.normal(0, 0.3, 3)
        q = 0.3 * np.sin(2 * math.pi * 1.2 * t[i] + np.arange(8)) + rng.normal(0, 0.01, 8)
        yield Reading(g, gyro, 0.02 * t[i], q)


def cmd_bench(a):
    r = PolicyRunner(a.run)
    readings = list(synthetic_readings(a.ticks))
    goals, ms = [], []
    for i, rd in enumerate(readings):
        cmd = 0.0 if i < a.ticks // 4 else 0.15
        t0 = time.perf_counter()
        goals.append(r.step(rd, cmd))
        ms.append((time.perf_counter() - t0) * 1000)
    ms = np.array(ms[10:])
    print(f"policy {a.run}: {r.obs_dim} inputs, {len(r.policy.layers)} layers, control {1 / r.dt:.0f} Hz")
    print("runner.step:", _stats(ms), f" -> {ms.mean() / (r.dt * 1000) * 100:.2f}% of the {r.dt * 1000:.0f} ms tick")
    if a.dump:
        np.save(a.dump, np.array(goals, dtype=np.float32))
        print("goals written to", a.dump)


def cmd_compare(a):
    from common.robot_config import RobotConfig

    res = math.radians(RobotConfig.load(ROBOT_CONFIG).actuator_spec["position_accuracy_deg"])
    x, y = np.load(a.a), np.load(a.b)
    tx, ty = np.rint(x / res).astype(int), np.rint(y / res).astype(int)
    diff = tx != ty
    print(f"{x.shape[0]} ticks x {x.shape[1]} servos: {diff.sum()} goals differ "
          f"(max {np.abs(tx - ty).max()} servo ticks), max |rad diff| {np.abs(x - y).max():.2e}")
    if diff.any():
        print("first difference at tick", int(np.argwhere(diff)[0][0]))


# ------------------------------------------------------------------- dryrun
def cmd_dryrun(a):
    from common.robot_config import RobotConfig
    from pi_runner.hardware import HardwareBackend

    r = PolicyRunner(a.run)
    rc = RobotConfig.load(ROBOT_CONFIG)
    hw = HardwareBackend(r.act_names, rc.actuator_spec["baud_rate"],
                         [rc.joint_range_rad(n) for n in r.act_names], servos="read_only" if a.bus else False)
    first = hw.read()
    r.reset_state(np.zeros(len(r.act_names)), yaw_ref=first.yaw)
    n = int(a.seconds / r.dt)
    period, compute, late = [], [], 0
    t_next = time.perf_counter()
    t_prev = None
    print(f"dry run: real IMU, {'real servo reads + moves to an absent id' if a.bus else 'servos untouched'}, {n} ticks at {1 / r.dt:.0f} Hz, cmd {a.cmd} m/s after 2 s")
    for k in range(n):
        now = time.perf_counter()
        if t_prev is not None:
            period.append((now - t_prev) * 1000)
        t_prev = now
        rd = hw.read()
        goal = r.step(rd, a.cmd if k * r.dt >= 2.0 else 0.0)
        hw.write(goal, r.dt)
        compute.append((time.perf_counter() - now) * 1000)
        if k % int(1 / r.dt) == 0:
            print(f"t {k * r.dt:5.1f} s  g_z {rd.gravity_b[2]:+.3f}  goal[deg] "
                  + " ".join(f"{math.degrees(v):+6.1f}" for v in goal)
                  + (f"  SAFETY: {r.safety.tripped}" if r.safety.tripped else ""), flush=True)
        t_next += r.dt
        slack = t_next - time.perf_counter()
        if slack > 0:
            time.sleep(slack)
        else:
            late += 1
            t_next = time.perf_counter()
    period, compute = np.array(period), np.array(compute)
    print(f"\nloop period:  {_stats(period)}  (target {r.dt * 1000:.0f} ms), overruns {late}")
    print(f"tick work:    {_stats(compute)}  (IMU read{' + servo reads' if a.bus else ''} + runner.step + write)")
    if a.bus:
        print(f"bus: {hw.bus.errors} missing replies re-read")
    print(f"I2C retries {hw.imu.errors}; safety {'tripped: ' + r.safety.tripped if r.safety.tripped else 'ok'}")
    hw.imu.close()
    if hw.bus is not None:
        hw.bus.close()


# ---------------------------------------------------------------------- bus
def cmd_bus(a):
    from common.robot_config import RobotConfig
    from pi_runner.hardware import load_hardware_config
    from pi_runner.lx16a import LX16ABus

    hw = load_hardware_config()
    bus = LX16ABus(hw["servo_bus"]["port"], RobotConfig.load(ROBOT_CONFIG).actuator_spec["baud_rate"],
                   hw["servo_bus"]["read_timeout_s"])
    print("adapter", bus.port)
    if a.scan:
        t0 = time.time()
        ids = bus.scan()
        print(f"servos answering: {ids} (scan {time.time() - t0:.1f} s)")
    else:
        ids = [s["id"] for s in hw["servos"].values() if s["id"] is not None]
        if not ids:
            print("no ids in hardware.yaml yet; run with --scan")
    for i in ids:
        print(f"  id {i:3d}: pos {bus.read_pos(i)} ticks, {bus.read_vin_v(i)} V, {bus.read_temp_c(i)} C")
    if ids:
        ms = []
        for _ in range(50):
            t0 = time.perf_counter()
            for i in ids:
                bus.read_pos(i)
            ms.append((time.perf_counter() - t0) * 1000)
        print(f"reading all {len(ids)} positions:", _stats(np.array(ms)), f"; reply errors {bus.errors}")
    bus.close()


# -------------------------------------------------------------------- watch
def cmd_watch(a):
    """Record all servo positions (read-only, torque untouched) for hand-moved joint tests."""
    from common.robot_config import RobotConfig
    from pi_runner.hardware import load_hardware_config
    from pi_runner.lx16a import LX16ABus

    hw = load_hardware_config()
    names = [n for n, c in hw["servos"].items() if c["id"] is not None]
    ids = [hw["servos"][n]["id"] for n in names]
    bus = LX16ABus(hw["servo_bus"]["port"], RobotConfig.load(ROBOT_CONFIG).actuator_spec["baud_rate"],
                   hw["servo_bus"]["read_timeout_s"])
    rows, t0, last = [], time.perf_counter(), 0.0
    while time.perf_counter() - t0 < a.seconds:
        p = [bus.read_pos(i) for i in ids]
        t = time.perf_counter() - t0
        rows.append([t] + [np.nan if v is None else v for v in p])
        if t - last > 1.0:
            last = t
            print(f"{t:5.1f} s " + " ".join(f"{n.replace('_joint', '')}={v}" for n, v in zip(names, p)), flush=True)
    bus.close()
    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, np.array(rows, dtype=float))
    print(f"{len(rows)} samples of {names} written to {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("info")
    s = sub.add_parser("imu")
    s.add_argument("--seconds", type=float, default=10)
    s = sub.add_parser("bench")
    s.add_argument("run")
    s.add_argument("--ticks", type=int, default=3000)
    s.add_argument("--dump")
    s = sub.add_parser("compare")
    s.add_argument("a")
    s.add_argument("b")
    s = sub.add_parser("dryrun")
    s.add_argument("run")
    s.add_argument("--seconds", type=float, default=20)
    s.add_argument("--cmd", type=float, default=0.10)
    s.add_argument("--bus", action="store_true", help="real servo reads, moves sent to an absent id")
    s = sub.add_parser("watch")
    s.add_argument("--seconds", type=float, default=60)
    s.add_argument("--out", default="~/trex/logs/servo_watch.npy")
    s = sub.add_parser("bus")
    s.add_argument("--scan", action="store_true")
    a = p.parse_args()
    {"info": cmd_info, "imu": cmd_imu, "bench": cmd_bench, "compare": cmd_compare, "dryrun": cmd_dryrun,
     "bus": cmd_bus, "watch": cmd_watch}[a.command](a)


if __name__ == "__main__":
    main()
