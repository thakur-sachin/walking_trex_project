"""Fit the servo supply sag of the BAM model (Isaac Lab BamActuatorCfg vin_drop_gain) to real walk logs:

    vin_eff = vin_rest - sag_gain * sum_j |tau_motor_j|

tau_motor is the LX-16A model's motor torque (walk_rl/actuator.py without friction, proportional to the motor
current), evaluated from the LOGGED servo goals and measured angles, so no simulation is involved:
    q_ref = previous goal -> goal ramp at the time of the voltage read, e = q_ref - q, duty = clamp(kp e, -1, 1),
    tau_motor = duty * stall - stall / no_load * qd     (stall, no_load at the resting battery voltage)
walk_live.py (2026-10-24 on) reads one servo's supply voltage per tick after the goal write, round robin.

    python3 lab3/fit_supply_sag.py assets/robots/trex/servo_logs/walk_v9v_20261024_090825_hang.npz ...
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.robot_config import RobotConfig  # noqa: E402
from walk_rl.actuator import LX16AParams  # noqa: E402

READ_FRAC = 0.6  # the voltage read happens ~25 of 40 ms into the tick (after ~20 ms of work)


def motor_torque(log: np.lib.npyio.NpzFile, p: LX16AParams, v_rest: float) -> np.ndarray:
    """(T, J) motor torque at the time of each tick's voltage read."""
    cols = list(log["cols"])
    rows = log["rows"]
    meta = json.loads(str(log["meta"]))
    dt = meta["dt"]
    js = meta["joints"]
    q = rows[:, [cols.index(f"q_{j}") for j in js]]
    goal = rows[:, [cols.index(f"goal_{j}") for j in js]]
    q_next = np.vstack([q[1:], q[-1:]])
    qd = (q_next - q) / dt
    q_now = q + READ_FRAC * (q_next - q)
    g_prev = np.vstack([goal[:1], goal[:-1]])
    q_ref = g_prev + READ_FRAC * (goal - g_prev)
    e = q_ref - q_now
    e = np.sign(e) * np.maximum(np.abs(e) - p.deadband, 0.0)
    duty = np.clip(p.kp * e, -1.0, 1.0)
    stall = p.stall * v_rest / p.v_ref
    no_load = p.no_load * v_rest / p.v_ref
    return duty * stall - stall / no_load * qd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=Path)
    a = ap.parse_args()
    p = LX16AParams.from_robot_config(RobotConfig.load(PROJECT_ROOT / "assets" / "robots" / "trex" / "robot_config.yaml"))
    xs, ys = [], []
    for f in a.logs:
        d = np.load(f)
        meta = json.loads(str(d["meta"]))
        ids = meta["ids"]
        v_rest = dict(zip(ids, meta["vin_before"]))
        vin = d["vin"]
        vin = vin[~np.isnan(vin[:, 2])]
        tau = motor_torque(d, p, float(np.mean(meta["vin_before"])))
        load = np.abs(tau).sum(axis=1)
        k = vin[:, 0].astype(int)
        drop = np.array([v_rest[int(s)] for s in vin[:, 1]]) - vin[:, 2]
        x = load[k]
        g, c = np.polyfit(x, drop, 1)
        r = np.corrcoef(x, drop)[0, 1]
        print(f"{f.name}: {len(x)} reads, sum|tau| mean {x.mean():.2f} max {x.max():.2f} N m, drop mean {drop.mean():.2f} "
              f"max {drop.max():.2f} V | fit drop = {g:.3f} V/(N m) * sum|tau| {c:+.3f} V, r {r:.2f}")
        xs.append(x)
        ys.append(drop)
    x, y = np.concatenate(xs), np.concatenate(ys)
    g0 = float((x * y).sum() / (x * x).sum())  # through the origin, as BAM's model
    g, c = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    q = np.percentile(y / np.maximum(x, 1e-3), [10, 50, 90])
    print(f"all: {len(x)} reads | through origin: sag_gain {g0:.3f} V/(N m) | affine {g:.3f} V/(N m) {c:+.3f} V, r {r:.2f} | "
          f"drop/load ratio p10/p50/p90 {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}")


if __name__ == "__main__":
    main()
