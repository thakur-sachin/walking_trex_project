"""BNO055 over I2C (smbus2): just what the policy needs, read in one 32-byte burst.

Register map and scales from the Bosch BNO055 datasheet (rev 1.4): gyro 0x14 (900 LSB per
rad/s with UNIT_SEL bit 1), quaternion 0x20 (2^14 LSB), gravity vector 0x2E (100 LSB per
m/s^2). Fusion output runs at 100 Hz, so a 25 Hz loop always gets a fresh sample.

Frames: the chip's axes are rotated into the body frame (x forward, y left, z up) with
`sensor_to_body` from hardware.yaml. The fused world frame has z up. The chip's gravity
vector points up when the chip is level (it is the accelerometer's reaction reading), while
the policy wants the unit direction gravity pulls in (level = (0, 0, -1)), hence the sign.
"""

from __future__ import annotations

import math
import time

import numpy as np

CHIP_ID, CHIP_ID_VALUE = 0x00, 0xA0
PAGE_ID = 0x07
DATA_START = 0x14  # GYR_DATA_X_LSB .. GRV_DATA_Z_MSB = 0x14..0x33, 32 bytes
CALIB_STAT = 0x35
SYS_STATUS, SYS_ERR = 0x39, 0x3A
UNIT_SEL = 0x3B
OPR_MODE = 0x3D
PWR_MODE = 0x3E
SYS_TRIGGER = 0x3F
MODES = {"CONFIG": 0x00, "IMUPLUS": 0x08, "NDOF": 0x0C}


class BNO055:
    def __init__(self, bus: int = 1, address: int = 0x28, mode: str = "IMUPLUS",
                 sensor_to_body=None):
        from smbus2 import SMBus  # Pi only

        self.bus = SMBus(bus)
        self.addr = address
        self.mode = mode
        self.R = np.eye(3) if sensor_to_body is None else np.asarray(sensor_to_body, dtype=float)
        assert np.allclose(self.R @ self.R.T, np.eye(3)) and np.isclose(np.linalg.det(self.R), 1.0), \
            "sensor_to_body must be a proper rotation (orthonormal, det +1)"
        self.errors = 0
        self._begin()

    @classmethod
    def from_config(cls, imu_cfg: dict) -> "BNO055":
        return cls(int(imu_cfg["bus"]), int(imu_cfg["address"]), imu_cfg["mode"], imu_cfg["sensor_to_body"])

    # ---------------------------------------------------------------- low level
    def _w(self, reg: int, val: int) -> None:
        self.bus.write_byte_data(self.addr, reg, val)

    def _r(self, reg: int) -> int:
        return self.bus.read_byte_data(self.addr, reg)

    def _begin(self) -> None:
        chip = self._r(CHIP_ID)
        if chip != CHIP_ID_VALUE:  # the chip needs ~650 ms after power-on
            time.sleep(0.7)
            chip = self._r(CHIP_ID)
        if chip != CHIP_ID_VALUE:
            raise RuntimeError(f"BNO055 chip id 0x{chip:02x}, expected 0xa0")
        self._w(PAGE_ID, 0)
        self._w(OPR_MODE, MODES["CONFIG"])
        time.sleep(0.025)
        self._w(PWR_MODE, 0x00)  # normal
        self._w(UNIT_SEL, 0x06)  # accel m/s^2, gyro rad/s, euler rad, Windows orientation
        self._w(SYS_TRIGGER, 0x00)  # internal oscillator
        self._w(OPR_MODE, MODES[self.mode])
        time.sleep(0.02)
        # Fusion outputs read all-zero for a few samples after the mode switch.
        deadline = time.time() + 1.0
        while np.linalg.norm(self.read_raw()[2]) < 5.0:
            if time.time() > deadline:
                raise RuntimeError("BNO055 fusion gave no gravity vector within 1 s")
            time.sleep(0.01)

    # --------------------------------------------------------------------- read
    def read_raw(self):
        """(gyro_chip rad/s, quaternion wxyz, gravity_chip m/s^2) from one burst read."""
        b = bytes(self.bus.read_i2c_block_data(self.addr, DATA_START, 32))
        v = np.frombuffer(b, dtype="<i2").astype(float)
        gyro = v[0:3] / 900.0
        quat = v[6:10] / 16384.0  # 0x20: w, x, y, z
        grav = v[13:16] / 100.0  # 0x2E
        return gyro, quat, grav

    def read(self):
        """Body-frame (gravity_b unit vector, gyro_b rad/s, yaw rad, CCW positive about z up).
        Retries an I2C read error once; raises on the second."""
        try:
            gyro, quat, grav = self.read_raw()
        except OSError:
            self.errors += 1
            gyro, quat, grav = self.read_raw()
        gyro_b = self.R @ gyro
        g = -(self.R @ grav)
        n = np.linalg.norm(g)
        gravity_b = g / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
        # Body x axis in the world frame: R_world_chip @ R^T @ e_x.
        w, x, y, z = quat
        Rwc = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
        fwd = Rwc @ self.R.T @ np.array([1.0, 0.0, 0.0])
        yaw = math.atan2(fwd[1], fwd[0])
        return gravity_b, gyro_b, yaw

    def calibration(self) -> dict:
        c = self._r(CALIB_STAT)
        return {"sys": c >> 6 & 3, "gyro": c >> 4 & 3, "accel": c >> 2 & 3, "mag": c & 3}

    def status(self) -> tuple[int, int]:
        return self._r(SYS_STATUS), self._r(SYS_ERR)

    def close(self) -> None:
        self.bus.close()
