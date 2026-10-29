"""Rank finished runs on the sim checks, to pick the policies for the real robot.

    python3 lab3/rank_runs.py v9s v9w n1 n2 ...

Per run: the gait scorecard (walk_rl/gait_scorecard.py) pass count, and the numbers that matter on the wood
floor (measured static mu 0.27): MuJoCo speed at 0.10 at mu 0.3-0.8 (median) and at mu 0.15 / 0.20 / 0.27
(worst of the three medians), falls on both grids, mean joint speed (servo current: the USB adapter resets),
tail speed, foot lift, mean |yaw| over 60 s, Isaac (PhysX) hard survival.
Order: no MuJoCo fall first, then the scorecard pass count, then the low-friction speed.
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "walk_rl"))
from gait_scorecard import RUNS, score  # noqa: E402


def grid(run: str, name: str):
    f = RUNS / run / name
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return d["rows"] if isinstance(d, dict) else d


def at(rows, cmd, mu=None):
    return [r for r in rows if abs(r["cmd"] - cmd) < 1e-6 and (mu is None or abs(r["friction"] - mu) < 1e-6)]


def summary(run: str) -> dict | None:
    try:
        sc = score(run)
    except FileNotFoundError:
        return None
    rows = grid(run, "sim2sim_policy_numpy_lx16a.json")
    low = grid(run, "sim2sim_lowmu_lx16a.json")
    ev = json.loads((RUNS / run / "eval_checkpoint_lx16a.json").read_text())
    r10 = at(rows, 0.10)
    out = {"run": run, "passed": sum(ok for *_, ok in sc), "of": len(sc),
           "mj_speed": st.median(r["vx"] / 0.10 for r in r10),
           "falls": sum(r["fell"] is not None and r["fell"] is not False for r in rows),
           "yaw": st.mean(abs(r["yaw_deg"]) for r in r10)}
    js = [r["joint_speed_deg_s"] for r in r10 if r.get("joint_speed_deg_s")]
    out["joint_speed"] = st.mean(st.mean(j.values()) for j in js) if js else None
    out["tail_speed"] = st.mean(j["tail"] for j in js) if js else None
    out["foot_lift"] = st.median(min(r["foot_lift_mm"]) for r in r10) if "foot_lift_mm" in r10[0] else None
    if low:
        mus = sorted({r["friction"] for r in low})
        out["low_speed"] = min(st.median(r["vx"] / 0.10 for r in at(low, 0.10, m)) for m in mus)
        out["falls"] += sum(r["fell"] is not None and r["fell"] is not False for r in low)
    else:
        out["low_speed"] = None
    hard = ev.get("hard") or {}
    out["hard_surv"] = min(v["survival"] for v in hard.values()) if isinstance(hard, dict) and hard and \
        isinstance(next(iter(hard.values())), dict) and "survival" in next(iter(hard.values())) else None
    for name, val, _ in sc:
        if name.startswith("Isaac hard survival"):
            out["hard_surv"] = float(val)
    return out


def main():
    res = [s for s in (summary(r) for r in sys.argv[1:]) if s]
    res.sort(key=lambda s: (s["falls"] > 0, -s["passed"], -(s["low_speed"] or 0)))
    f = lambda v, fmt: "-" if v is None else format(v, fmt)
    print("| rank | run | scorecard | MuJoCo @0.10 | low mu @0.10 (worst) | falls | joint speed | tail | foot lift | |yaw| 60 s | PhysX hard |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, s in enumerate(res, 1):
        print(f"| {i} | {s['run']} | {s['passed']}/{s['of']} | {f(100 * s['mj_speed'], '.0f')} % | "
              f"{'-' if s['low_speed'] is None else format(100 * s['low_speed'], '.0f') + ' %'} | {s['falls']} | "
              f"{f(s['joint_speed'], '.0f')} deg/s | {f(s['tail_speed'], '.0f')} deg/s | {f(s['foot_lift'], '.0f')} mm | "
              f"{f(s['yaw'], '.1f')} deg | {f(s['hard_surv'], '.2f')} |")


if __name__ == "__main__":
    main()
