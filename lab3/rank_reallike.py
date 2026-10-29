"""Table of the real-like MuJoCo grid (lab3/rescore_reallike.sh), worst cases first.

    python3 lab3/rank_reallike.py v9o v9s n21 ...   [--policy-suffix _best_numpy]

Per run, at the 0.10 m/s command: speed in % of the command (median and worst cell over friction 0.2 / 0.27 / 0.4,
bus delay 0-1, supply 6.0 / 6.7 V, IMU lag 1-2), the median at wood friction 0.27, falls (all cells), mean |yaw|
after 30 s, stance-sole slide per metre forward, swing clearance of the lowest sole point, the share of time a
servo is above the datasheet stall, the mean total servo torque (current proxy), and the Isaac (PhysX) speed at 0.10 and
hard survival (the only engine a Newton-trained policy never saw). Order: falls, then the worst real-like cell.
"""
import json
import statistics as st
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "outputs" / "walk_runs"


def summary(run: str, suffix: str = "") -> dict | None:
    f = RUNS / run / f"sim2sim_reallike{suffix}.json"
    if not f.exists():
        return None
    rows = json.loads(f.read_text())["rows"]
    r10 = [r for r in rows if abs(r["cmd"] - 0.10) < 1e-6]
    sp = [r["vx"] / 0.10 for r in r10]
    m = lambda k: st.median(r[k] for r in r10 if k in r)
    ev = RUNS / run / "eval_checkpoint_lx16a.json"
    px, hard = None, None
    if ev.exists():
        e = json.loads(ev.read_text())
        px = next((r["vx"] / 0.10 for r in e.get("nominal", []) if abs(r["cmd"] - 0.10) < 1e-6), None)
        hard = min((r["survive"] for r in e.get("hard", [])), default=None)
    return {"physx": px, "hard": hard, "run": run + suffix, "med": st.median(sp), "worst": min(sp),
            "wood": st.median(r["vx"] / 0.10 for r in r10 if abs(r["friction"] - 0.27) < 1e-6),
            "falls": sum(bool(r["fell"]) for r in rows), "n": len(rows),
            "yaw": st.mean(abs(r["yaw_deg"]) for r in r10), "slide": m("slide_per_forward"),
            "clear": m("swing_clearance_mm"), "over": m("over_stall_share"), "tau": m("tau_sum_mean_nm")}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    suffix = sys.argv[sys.argv.index("--policy-suffix") + 1] if "--policy-suffix" in sys.argv else ""
    args = [a for a in args if a != suffix]
    res = [s for s in (summary(r, suffix) for r in args) if s]
    res.sort(key=lambda s: (s["falls"] > 0, -s["worst"]))
    print("| run | speed @0.10 median | worst | wood mu 0.27 | falls | abs yaw 30 s | slide per m | swing clearance | over stall | sum torque | PhysX @0.10 | PhysX hard |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in res:
        print(f"| {s['run']} | {100 * s['med']:.0f} % | {100 * s['worst']:.0f} % | {100 * s['wood']:.0f} % | {s['falls']}/{s['n']} | "
              f"{s['yaw']:.0f} deg | {s['slide']:.2f} | {s['clear']:.1f} mm | {100 * s['over']:.0f} % | {s['tau']:.1f} N m | "
              f"{'-' if s['physx'] is None else format(100 * s['physx'], '.0f') + ' %'} | {'-' if s['hard'] is None else format(s['hard'], '.2f')} |")


if __name__ == "__main__":
    main()
