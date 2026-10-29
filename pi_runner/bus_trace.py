"""walk_live with every pipelined servo position read traced (servo bus diagnosis).

    python -m pi_runner.bus_trace RUN_DIR --schedule 0:3 0.10:60 0:3            # robot hanging
    GAP_S=0.003 python -m pi_runner.bus_trace RUN_DIR ...                        # other request gap

Same arguments and behaviour as pi_runner/walk_live.py. Per read it records when each request went out
and when each reply was parsed (ms after the first request; USB delivers replies in chunks, so the
arrival times are quantized), and on a miss the raw bytes received and the result of the single re-read.
Trace: ~/trex/logs/bus_trace_<time>.json, next to the walk log.

Result 2026-10-23 (v9s hanging, 3 x 1651 reads at 2.0 / 2.5 / 3.0 ms gap): most misses are one servo
that does not answer for > 10 ms (request and re-read) while every other reply is clean; they grow toward
the far end of each daisy chain (tibias >> femurs > hips) and do not go away with a longer gap. A rarer
kind: a request sent > 1 ms late by the Pi corrupts the replies around it.
"""
import atexit
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pi_runner.lx16a as lx  # noqa: E402

TRACE = []
_take = lx.LX16ABus._take_pos_replies


def read_positions(self, ids, gap_s=0.002, deadline_s=0.025):
    """lx16a.LX16ABus.read_positions with timing and raw-byte capture."""
    ids = list(ids)
    gap_s = float(os.environ.get("GAP_S", gap_s))
    deadline_s = max(deadline_s, (len(ids) - 1) * gap_s + 0.011)  # same reply margin as 2.0 ms x 7 + 11 = 25
    self.wait_tx()
    self.ser.reset_input_buffer()
    t0 = time.perf_counter()
    buf, raw, sent, arr, got = b"", b"", {}, {}, {}

    def take(b):
        before = set(got)
        b = _take(b, got)
        for sid in set(got) - before:
            arr[sid] = (time.perf_counter() - t0) * 1000
        return b

    for n, sid in enumerate(ids):
        while time.perf_counter() - t0 < n * gap_s:
            if self.ser.in_waiting:
                chunk = self.ser.read(self.ser.in_waiting)
                raw += chunk
                buf = take(buf + chunk)
        sent[sid] = (time.perf_counter() - t0) * 1000
        self._send(sid, lx.SERVO_POS_READ)
    while len(got) < len(ids) and time.perf_counter() - t0 < deadline_s:
        chunk = self.ser.read(max(1, self.ser.in_waiting))
        raw += chunk
        buf = take(buf + chunk)
    rec = {"gap_ms": gap_s * 1000, "t": time.time(), "sent": sent, "arr": arr}
    missing = [sid for sid in ids if sid not in got]
    if missing:
        self.errors += len(missing)
        rec |= {"missing": missing, "raw": raw.hex(), "retry": {}}
        for sid in missing:
            t = time.perf_counter()
            p = self.read_pos(sid)
            rec["retry"][sid] = [p, (time.perf_counter() - t) * 1000]
            if p is not None:
                got[sid] = p
    TRACE.append(rec)
    return [got.get(sid) for sid in ids]


@atexit.register
def _dump():
    if not TRACE:
        return
    out = Path.home() / "trex" / "logs" / f"bus_trace_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(TRACE))
    print(f"trace: {out} ({len(TRACE)} reads, {sum('missing' in r for r in TRACE)} with a first-pass miss)")


if __name__ == "__main__":
    lx.LX16ABus.read_positions = read_positions
    from pi_runner import walk_live
    walk_live.main()
