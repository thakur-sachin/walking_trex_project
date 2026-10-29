"""LX-16A serial bus servos through the Hiwonder/LewanSoul BusLinker USB adapter (pyserial).

Protocol (LX-16A bus servo communication protocol): frame = 0x55 0x55 id len cmd params
checksum, len = len(params) + 3, checksum = ~(id + len + cmd + sum(params)) & 0xFF. Position
is in ticks, 0-1000 over 0-240 deg (0.24 deg/tick); move time in ms. The bus is half-duplex:
depending on the adapter the request can come back as an echo before the reply, so replies
are matched by id, command and length and anything else is skipped.

There is no synchronous multi-servo write: 8 MOVE_TIME_WRITE frames are 80 bytes = 6.9 ms at
115200 baud, sent as one USB transfer (`move_many`). Position reads are pipelined
(`read_positions`): one POS_READ round trip is 3.6 ms through the BusLinker (CH340), but the
servo answers ~1.7 ms after its request and the rest is USB latency, so requests go out every
`request_gap_s` and the replies come back staggered. Measured 2026-10-18 (8 servos, 300 runs per
gap): 1.6 ms gap loses replies in 8/300, 1.7 ms in 1/300, 1.8-2.2 ms in 0/300; 2.0 ms reads all
8 in 15.7 ms (max 15.9) instead of 29 ms one by one.
"""

from __future__ import annotations

import glob
import time

SERVO_MOVE_TIME_WRITE = 1
SERVO_MOVE_STOP = 12
SERVO_ID_READ = 14
SERVO_ANGLE_OFFSET_READ = 19
SERVO_ANGLE_LIMIT_WRITE = 20
SERVO_ANGLE_LIMIT_READ = 21
SERVO_VIN_LIMIT_READ = 23
SERVO_TEMP_READ = 26
SERVO_VIN_READ = 27
SERVO_POS_READ = 28
SERVO_OR_MOTOR_MODE_READ = 30
SERVO_LOAD_OR_UNLOAD_WRITE = 31
SERVO_LOAD_OR_UNLOAD_READ = 32
BROADCAST_ID = 254
TICKS_PER_DEG = 1000.0 / 240.0
REPLY_PARAMS = {SERVO_ID_READ: 1, SERVO_ANGLE_OFFSET_READ: 1, SERVO_ANGLE_LIMIT_READ: 4, SERVO_VIN_LIMIT_READ: 4,
                SERVO_TEMP_READ: 1, SERVO_VIN_READ: 2, SERVO_POS_READ: 2, SERVO_OR_MOTOR_MODE_READ: 4,
                SERVO_LOAD_OR_UNLOAD_READ: 1}


def find_port(port: str = "auto") -> str:
    if port != "auto":
        return port
    ports = sorted(glob.glob("/dev/serial/by-id/*"))
    if len(ports) != 1:
        raise RuntimeError(f"servo_bus.port is auto but {len(ports)} USB serial devices are plugged in: {ports}. "
                           "Plug in the BusLinker only, or set the by-id path in hardware.yaml.")
    return ports[0]


def frame(sid: int, cmd: int, params: bytes = b"") -> bytes:
    ln = len(params) + 3
    chk = ~(sid + ln + cmd + sum(params)) & 0xFF
    return bytes([0x55, 0x55, sid, ln, cmd]) + params + bytes([chk])


class LX16ABus:
    def __init__(self, port: str = "auto", baud: int = 115200, read_timeout_s: float = 0.004):
        import serial  # Pi only

        self.port = find_port(port)
        self.ser = serial.Serial(self.port, baud, timeout=read_timeout_s)
        self.timeout = read_timeout_s
        self.errors = 0
        self.byte_s = 10.0 / baud  # 8N1
        self.tx_done = 0.0  # perf_counter time by which everything written is on the wire

    def close(self) -> None:
        self.ser.close()

    # ---------------------------------------------------------------- low level
    def _write(self, data: bytes) -> None:
        now = time.perf_counter()
        self.tx_done = max(self.tx_done, now) + len(data) * self.byte_s
        self.ser.write(data)

    def _send(self, sid: int, cmd: int, params: bytes = b"") -> None:
        self._write(frame(sid, cmd, params))

    def wait_tx(self, margin_s: float = 0.002) -> None:
        """Block until earlier writes have left the adapter (plus USB latency margin)."""
        while time.perf_counter() < self.tx_done + margin_s:
            pass

    def _reply(self, sid: int, cmd: int) -> bytes | None:
        """Params of the reply to `cmd` from `sid`, or None on timeout / bad checksum."""
        want = REPLY_PARAMS[cmd] + 3
        deadline = time.perf_counter() + self.timeout + 0.002
        buf = b""
        need = want + 3  # a whole reply frame; ser.read(n) blocks until n bytes or timeout,
        while time.perf_counter() < deadline:  # so never ask for more than the frame still lacks
            buf += self.ser.read(max(1, need))
            while True:
                i = buf.find(b"\x55\x55")
                if i < 0:
                    buf = buf[-1:]
                    need = 4 - len(buf)
                    break
                buf = buf[i:]
                if len(buf) < 4:
                    need = 4 - len(buf)
                    break
                ln = buf[3]
                if len(buf) < ln + 3:
                    need = ln + 3 - len(buf)
                    break
                need = 4
                f, buf = buf[: ln + 3], buf[ln + 3:]
                rid, rcmd, params, chk = f[2], f[4], f[5:-1], f[-1]
                if chk != (~(rid + ln + rcmd + sum(params)) & 0xFF):
                    continue
                if rcmd == cmd and ln == want and (rid == sid or sid == BROADCAST_ID):
                    return params
                # anything else (echo of our request, stale reply) is skipped
        self.errors += 1
        return None

    def _query(self, sid: int, cmd: int) -> bytes | None:
        self.wait_tx()
        self.ser.reset_input_buffer()
        self._send(sid, cmd)
        return self._reply(sid, cmd)

    # ------------------------------------------------------------------- reads
    def read_pos(self, sid: int) -> int | None:
        p = self._query(sid, SERVO_POS_READ)
        return None if p is None else int.from_bytes(p, "little", signed=True)

    def read_vin_v(self, sid: int) -> float | None:
        p = self._query(sid, SERVO_VIN_READ)
        return None if p is None else int.from_bytes(p, "little") / 1000.0

    def read_temp_c(self, sid: int) -> int | None:
        p = self._query(sid, SERVO_TEMP_READ)
        return None if p is None else p[0]

    def read_id(self, sid: int = BROADCAST_ID) -> int | None:
        """With BROADCAST_ID only valid when a single servo is on the bus."""
        p = self._query(sid, SERVO_ID_READ)
        return None if p is None else p[0]

    def read_settings(self, sid: int) -> dict | None:
        """Stored configuration (read-only): angle offset, angle and voltage limits, mode, torque."""
        off = self._query(sid, SERVO_ANGLE_OFFSET_READ)
        lim = self._query(sid, SERVO_ANGLE_LIMIT_READ)
        vlim = self._query(sid, SERVO_VIN_LIMIT_READ)
        mode = self._query(sid, SERVO_OR_MOTOR_MODE_READ)
        load = self._query(sid, SERVO_LOAD_OR_UNLOAD_READ)
        if None in (off, lim, vlim, mode, load):
            return None
        u16 = lambda b, i: int.from_bytes(b[i:i + 2], "little")
        return {"angle_offset_ticks": int.from_bytes(off, "little", signed=True),
                "angle_limit_ticks": [u16(lim, 0), u16(lim, 2)],
                "vin_limit_v": [u16(vlim, 0) / 1000, u16(vlim, 2) / 1000],
                "motor_mode": mode[0] == 1, "torque_on": load[0] == 1}

    def read_positions(self, ids, gap_s: float = 0.002, deadline_s: float = 0.025) -> list[int | None]:
        """Pipelined POS_READ of several servos (see the module docstring). A missing reply is
        re-read on its own; still missing -> None."""
        ids = list(ids)
        self.wait_tx()  # queued move commands would otherwise bunch the requests up
        self.ser.reset_input_buffer()
        t0 = time.perf_counter()
        buf = b""
        for n, sid in enumerate(ids):
            while time.perf_counter() - t0 < n * gap_s:
                if self.ser.in_waiting:
                    buf += self.ser.read(self.ser.in_waiting)
            self._send(sid, SERVO_POS_READ)
        got: dict[int, int] = {}
        while len(got) < len(ids) and time.perf_counter() - t0 < deadline_s:
            buf += self.ser.read(max(1, self.ser.in_waiting))
            buf = self._take_pos_replies(buf, got)
        missing = [sid for sid in ids if sid not in got]
        if missing:
            self.errors += len(missing)
            for sid in missing:
                p = self.read_pos(sid)
                if p is not None:
                    got[sid] = p
        return [got.get(sid) for sid in ids]

    @staticmethod
    def _take_pos_replies(buf: bytes, got: dict) -> bytes:
        while True:
            i = buf.find(b"\x55\x55")
            if i < 0:
                return buf[-1:]
            buf = buf[i:]
            if len(buf) < 4 or len(buf) < buf[3] + 3:
                return buf
            ln = buf[3]
            f, rest = buf[: ln + 3], buf[ln + 3:]
            rid, cmd, params, chk = f[2], f[4], f[5:-1], f[-1]
            if chk == (~(rid + ln + cmd + sum(params)) & 0xFF) and cmd == SERVO_POS_READ and ln == 5:
                got[rid] = int.from_bytes(params, "little", signed=True)
                buf = rest
            else:
                buf = buf[2:]  # not a valid reply here: resync on the next header

    def move_many(self, targets, move_time_ms: int) -> None:
        """MOVE_TIME_WRITE for several (id, tick) pairs in one serial write."""
        ms = max(0, min(30000, int(move_time_ms))).to_bytes(2, "little")
        self._write(b"".join(frame(sid, SERVO_MOVE_TIME_WRITE, max(0, min(1000, int(t))).to_bytes(2, "little") + ms)
                                for sid, t in targets))

    def scan(self, ids=range(0, 254)) -> list[int]:
        return [i for i in ids if self.read_pos(i) is not None]

    # ------------------------------------------------------------------ writes
    def move(self, sid: int, tick: int, move_time_ms: int) -> None:
        tick = max(0, min(1000, int(tick)))
        ms = max(0, min(30000, int(move_time_ms)))
        self._send(sid, SERVO_MOVE_TIME_WRITE, tick.to_bytes(2, "little") + ms.to_bytes(2, "little"))

    def write_angle_limits(self, sid: int, lo: int, hi: int) -> None:
        """Stored (EEPROM, survives power-off) position limits; the servo clamps every move to them."""
        lo, hi = int(lo), int(hi)
        assert 0 <= lo < hi <= 1000, (lo, hi)
        self._send(sid, SERVO_ANGLE_LIMIT_WRITE, lo.to_bytes(2, "little") + hi.to_bytes(2, "little"))

    def set_torque(self, sid: int, on: bool) -> None:
        self._send(sid, SERVO_LOAD_OR_UNLOAD_WRITE, bytes([1 if on else 0]))

    def stop(self, sid: int) -> None:
        self._send(sid, SERVO_MOVE_STOP)
