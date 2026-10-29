"""Set each servo's stored angle limits (EEPROM) to its joint range from robot_config.yaml plus a
margin, mapped through hardware.yaml (id, dir, zero_tick). The servo then clamps every command to
the joint range by itself, independent of the Pi.

    python -m pi_runner.servo_limits            # show current vs proposed, change nothing
    python -m pi_runner.servo_limits --apply    # back up current limits, write, read back
    python -m pi_runner.servo_limits --restore FILE   # write back a backup
    python -m pi_runner.servo_limits --joints left_hip_joint right_hip_joint --margin-deg 1 --apply
                                                # only these servos, with a smaller margin (the hip
                                                # inward range ends 3 deg before a hard contact)
Nothing moves: only limits are written.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.robot_config import RobotConfig  # noqa: E402
from pi_runner.hardware import load_hardware_config  # noqa: E402
from pi_runner.lx16a import TICKS_PER_DEG, LX16ABus  # noqa: E402
from pi_runner.runner import ROBOT_CONFIG  # noqa: E402

MARGIN_DEG = 3.0


def proposed(rc: RobotConfig, hw: dict, joints: list[str], margin_deg: float = MARGIN_DEG) -> dict:
    out = {}
    for n in joints:
        c = hw["servos"][n]
        if None in (c["id"], c["dir"], c["zero_tick"]):
            raise SystemExit(f"{n}: id/dir/zero_tick not set in hardware.yaml")
        lo_rad, hi_rad = rc.joint_range_rad(n)
        ends = sorted(c["zero_tick"] + c["dir"] * (math.degrees(x) + s * margin_deg) * TICKS_PER_DEG
                      for x, s in ((lo_rad, -1), (hi_rad, 1)))
        out[n] = {"id": c["id"], "lo": max(0, math.floor(ends[0])), "hi": min(1000, math.ceil(ends[1]))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--restore")
    ap.add_argument("--joints", nargs="+", default=None, help="only these joints (default: all actuated)")
    ap.add_argument("--margin-deg", type=float, default=MARGIN_DEG)
    a = ap.parse_args()
    rc, hw = RobotConfig.load(ROBOT_CONFIG), load_hardware_config()
    bus = LX16ABus(hw["servo_bus"]["port"], rc.actuator_spec["baud_rate"], hw["servo_bus"]["read_timeout_s"])
    try:
        if a.restore:
            target = {k: {"id": int(v["id"]), "lo": v["lo"], "hi": v["hi"]} for k, v in json.loads(Path(a.restore).read_text()).items()}
        else:
            joints = a.joints or list(rc.actuated_joints)
            unknown = set(joints) - set(rc.actuated_joints)
            if unknown:
                raise SystemExit(f"not actuated joints: {sorted(unknown)}")
            target = proposed(rc, hw, joints, a.margin_deg)
        current = {}
        for n, t in target.items():
            s = None
            for _ in range(3):
                s = bus.read_settings(t["id"])
                if s:
                    break
            if s is None:
                raise SystemExit(f"{n} (id {t['id']}): no reply")
            current[n] = {"id": t["id"], "lo": s["angle_limit_ticks"][0], "hi": s["angle_limit_ticks"][1],
                          "pos": bus.read_pos(t["id"])}
            print(f"{n:18s} id {t['id']}: stored {current[n]['lo']:4d}..{current[n]['hi']:4d}  ->  "
                  f"{t['lo']:4d}..{t['hi']:4d}   (now at {current[n]['pos']})")
        if not (a.apply or a.restore):
            print("dry run: nothing written (use --apply)")
            return
        logs = Path.home() / "trex" / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        backup = logs / f"servo_limits_backup_{time.strftime('%Y%m%d_%H%M%S')}.json"
        backup.write_text(json.dumps(current, indent=1))
        print("backup:", backup)
        bad = []
        for n, t in target.items():
            bus.write_angle_limits(t["id"], t["lo"], t["hi"])
            time.sleep(0.02)
            s = None
            for _ in range(3):
                s = bus.read_settings(t["id"])
                if s:
                    break
            got = s and s["angle_limit_ticks"]
            ok = got == [t["lo"], t["hi"]]
            bad += [] if ok else [n]
            print(f"{n:18s} id {t['id']}: read back {got} {'ok' if ok else 'MISMATCH'}")
        print("all limits written and verified" if not bad else f"MISMATCH on {bad}")
    finally:
        bus.close()


if __name__ == "__main__":
    main()
