"""Real-robot backend for PolicyRunner: BNO055 + LX-16A bus, same read()/write() interface as
pi_runner/mujoco_backend.py. Wiring and calibration come from assets/robots/trex/hardware.yaml.

Servo ticks <-> joint radians: tick = zero_tick + dir * deg(q) * 1000/240.

`servos=False` is the dry run: the IMU is real, the servos are not touched, and the joint
reading is the last goal (an ideal servo). `servos="read_only"` adds the real bus traffic: real
position reads, and the move commands go out to an id no servo has, so nothing moves. `True` is
the deployment path.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

from pi_runner.bno055 import BNO055
from pi_runner.lx16a import TICKS_PER_DEG, LX16ABus
from pi_runner.runner import PROJECT_ROOT, Reading

DUMMY_ID = 200  # not on the bus (scan 2026-10-18 found ids 1-8 only)
HARDWARE_CONFIG = PROJECT_ROOT / "assets" / "robots" / "trex" / "hardware.yaml"


def load_hardware_config(path: str | Path = HARDWARE_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class HardwareBackend:
    def __init__(self, act_names: list[str], baud: int, joint_limits_rad, hw: dict | None = None,
                 servos: bool = True, max_missed_reads: int = 0):
        hw = hw or load_hardware_config()
        self.act_names = list(act_names)
        cal = hw["servos"]
        missing = [n for n in self.act_names if n not in cal]
        assert not missing, f"hardware.yaml has no servo entry for {missing}"
        self.ids = [cal[n]["id"] for n in self.act_names]
        self.read_order = list(range(len(self.ids)))  # bus slot -> joint index; set_read_order() changes it
        self.calibrated = all(cal[n][k] is not None for n in self.act_names for k in ("id", "dir", "zero_tick"))
        self.dir = np.array([cal[n]["dir"] or 0 for n in self.act_names], dtype=float)
        self.zero = np.array([cal[n]["zero_tick"] or 0 for n in self.act_names], dtype=float)
        assert not self.calibrated or set(self.dir) <= {-1.0, 1.0}, "servo dir must be +1 or -1"
        self.imu = BNO055.from_config(hw["imu"])
        self.servos = servos
        self.bus = None
        self.last_goal = np.zeros(len(self.act_names))
        self.gap_s = float(hw["servo_bus"]["request_gap_s"])
        self.write_ids = None
        # a servo whose position reply is missing (after the bus re-read) keeps its last good
        # position for up to max_missed_reads ticks in a row; one more raises
        self.max_missed_reads = int(max_missed_reads)
        self.last_q = None
        self.missed = np.zeros(len(self.act_names), dtype=int)  # consecutive misses per servo
        if servos:
            unset = [n for n in self.act_names if any(cal[n][k] is None for k in ("id", "dir", "zero_tick"))]
            if unset:
                raise RuntimeError(f"servo id/dir/zero_tick not set in hardware.yaml for {unset}; "
                                   "calibrate before driving servos (or use the dry run)")
            bad = []
            for k, n in enumerate(self.act_names):
                t = sorted(self.zero[k] + self.dir[k] * np.degrees(joint_limits_rad[k]) * TICKS_PER_DEG)
                if t[0] < 0 or t[1] > 1000:
                    bad.append(f"{n} needs ticks {t[0]:.0f}..{t[1]:.0f}")
            if bad:
                raise RuntimeError("joint range does not fit the servo's 0-1000 ticks (re-index the horn): "
                                   + "; ".join(bad))
            self.bus = LX16ABus(hw["servo_bus"]["port"], baud, hw["servo_bus"]["read_timeout_s"])
            if servos == "read_only":
                # bus-timing dry run: real position reads, moves addressed to an id nobody has,
                # so the bus carries exactly the deployment traffic and nothing moves
                assert DUMMY_ID not in self.ids
                self.write_ids = [DUMMY_ID] * len(self.ids)

    # ------------------------------------------------------------- conversions
    def rad_to_tick(self, q: np.ndarray) -> np.ndarray:
        return np.rint(self.zero + self.dir * np.degrees(q) * TICKS_PER_DEG)

    def tick_to_rad(self, t: np.ndarray) -> np.ndarray:
        return np.radians((np.asarray(t, dtype=float) - self.zero) * self.dir / TICKS_PER_DEG)

    # --------------------------------------------------------------- interface
    def read(self) -> Reading:
        gravity_b, gyro_b, yaw = self.imu.read()
        if self.bus is None:
            q = self.last_goal.copy()
        else:
            slot = self.bus.read_positions([self.ids[k] for k in self.read_order], self.gap_s)
            ticks = [None] * len(self.ids)
            for k, t in zip(self.read_order, slot):
                ticks[k] = t
            miss = np.array([t is None for t in ticks])
            self.missed = np.where(miss, self.missed + 1, 0)
            if miss.any() and (self.last_q is None or self.missed.max() > self.max_missed_reads):
                bad = [f"{n} x{k}" for n, k in zip(self.act_names, self.missed) if k]
                raise IOError(f"no position reply from {bad}")
            q = self.tick_to_rad([0 if t is None else t for t in ticks])
            if miss.any():
                q[miss] = self.last_q[miss]
            self.last_q = q
        return Reading(gravity_b, gyro_b, yaw, q)

    def set_read_order(self, order: str) -> None:
        """Order of the pipelined position reads: "joints" (act_names order) or "reversed" (bus diagnosis)."""
        n = len(self.ids)
        self.read_order = {"joints": list(range(n)), "reversed": list(range(n))[::-1]}[order]

    def write(self, goal: np.ndarray, move_time_s: float) -> None:
        self.last_goal = np.asarray(goal, dtype=float).copy()
        if self.bus is None:
            return
        ms = int(math.ceil(move_time_s * 1000))
        ids = self.ids if self.write_ids is None else self.write_ids
        self.bus.move_many(zip(ids, self.rad_to_tick(self.last_goal)), ms)

    def unload(self) -> None:
        """Torque off on every servo (the safety action)."""
        if self.bus is not None and self.write_ids is None:
            for sid in self.ids:
                self.bus.set_torque(sid, False)

    def close(self) -> None:
        self.unload()
        self.imu.close()
        if self.bus is not None:
            self.bus.close()
