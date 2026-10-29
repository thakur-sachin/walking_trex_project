"""Gait scorecard: one fixed set of targets every policy is judged on (plus its video).

    python walk_rl/gait_scorecard.py v9m v9n v9o

Reads outputs/walk_runs/<run>/eval_checkpoint_lx16a.json (walk_rl/evaluate.py --actuator lx16a)
and sim2sim_policy_numpy_lx16a.json (pi_runner/sim2sim.py --actuator lx16a, with gait statistics).
MuJoCo numbers are medians over the friction x delay grid at the stated command. No simulator
needed; prints one column per run and PASS/FAIL per target.

Targets (why):
- MuJoCo survival 100 %: an engine the policy never trained in stands in for the real robot.
- MuJoCo speed at 0.10 >= 70 %: Isaac-only speed does not transfer; first steps use 0.10.
- Isaac speed 0.05-0.15 within 80-120 %: the command means what it says.
- |left - right| femur and tibia <= 10 deg (MuJoCo 0.10): the legs are mirrored hardware, so a
  sustained difference is a learned limp (v9i: 72 deg tibia), not a need of the body.
- no joint within 3 deg of its range end > 20 % of the time (MuJoCo 0.10): a pinned joint has no
  margin to react (v9i: head and tail pinned).
- roll 5-95 % spread <= 12 deg (MuJoCo 0.10): roll is the hip servos' load and the tip-over margin.
- foot lift >= 12 mm on each foot (MuJoCo 0.10, 95th - 5th percentile foot height; added 2026-10-22): v9s
  shuffled at 6 mm, made 83-104 % in MuJoCo but 36 % on wood (> 70 % on carpet); median over the grid: v9o 10 mm,
  v9m 16 mm (on M5).
- sideways drift <= 30 % of forward distance (MuJoCo 0.10).
- both feet step forward, the shorter step >= half the longer (Isaac 0.10 and 0.15): v9i's left
  foot stepped backward at 0.10 (a scoot that PhysX rewards and MuJoCo does not).
- servo saturation <= 3 % and hard-sensor survival >= 0.95 (Isaac).
"""
import json
import statistics as st
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "outputs" / "walk_runs"


def load(run: str) -> tuple[dict, dict]:
    ev = json.loads((RUNS / run / "eval_checkpoint_lx16a.json").read_text())
    s2 = json.loads((RUNS / run / "sim2sim_policy_numpy_lx16a.json").read_text())
    return ev, s2


def score(run: str) -> list[tuple[str, str, bool]]:
    ev, s2 = load(run)
    rows = s2["rows"]
    at = lambda c: [r for r in rows if abs(r["cmd"] - c) < 1e-6]
    out = []
    surv = sum(r["fell"] is None for r in rows)
    out.append(("MuJoCo survival", f"{surv}/{len(rows)}", surv == len(rows)))
    r10 = at(0.10)
    sp = st.median(r["vx"] / 0.10 for r in r10)
    out.append(("MuJoCo speed @0.10", f"{100 * sp:.0f}%", sp >= 0.70))
    nom = {round(r["cmd"], 3): r for r in ev["nominal"]}
    pcts = [nom[c]["vx"] / c for c in (0.05, 0.10, 0.15) if c in nom]
    out.append(("Isaac speed 0.05-0.15", "/".join(f"{100 * p:.0f}" for p in pcts) + "%", all(0.8 <= p <= 1.2 for p in pcts)))
    if "femur_lr_deg" in r10[0]:
        fl = st.median(r["femur_lr_deg"] for r in r10)
        tl = st.median(r["tibia_lr_deg"] for r in r10)
        out.append(("femur / tibia L-R", f"{fl:+.0f} / {tl:+.0f} deg", abs(fl) <= 10 and abs(tl) <= 10))
        names = list(r10[0]["at_limit"])
        lim = {n: st.median(r["at_limit"][n] for r in r10) for n in names}
        worst = max(lim, key=lim.get)
        out.append(("worst joint at limit", f"{worst} {100 * lim[worst]:.0f}%", lim[worst] <= 0.20))
        roll = st.median(r["roll_spread_deg"] for r in r10)
        out.append(("roll spread @0.10", f"{roll:.0f} deg", roll <= 12))
        lift = min(st.median(r["foot_lift_mm"][k] for r in r10) for k in (0, 1))
        out.append(("foot lift min @0.10", f"{lift:.0f} mm", lift >= 12))
    else:
        out.append(("gait statistics", "rerun sim2sim", False))
    drift = st.median(abs(r["side_m"]) / max(abs(r["forward_m"]), 1e-3) for r in r10)
    out.append(("side / forward @0.10", f"{100 * drift:.0f}%", drift <= 0.30))
    steps = [(nom[c]["step_L"], nom[c]["step_R"]) for c in (0.10, 0.15) if c in nom]
    sym = all(min(a, b) > 0 and min(a, b) >= 0.5 * max(a, b) for a, b in steps)
    out.append(("Isaac step L/R @0.10,0.15", " ".join(f"{100 * a:.1f}/{100 * b:.1f}" for a, b in steps) + " cm", sym))
    walk = [r for r in ev["nominal"] if r["cmd"] > 0]
    sat = max(r["sat"] for r in walk)
    out.append(("Isaac saturation max", f"{100 * sat:.1f}%", sat <= 0.03))
    hs = min(r["survive"] for r in ev["hard"])
    out.append(("Isaac hard survival min", f"{hs:.2f}", hs >= 0.95))
    return out


if __name__ == "__main__":
    runs = sys.argv[1:]
    cards = {r: score(r) for r in runs}
    labels = [k for k, _, _ in cards[runs[0]]]
    w = max(len(k) for k in labels)
    print("target".ljust(w) + "".join(f" | {r:>18s}" for r in runs))
    for i, k in enumerate(labels):
        cells = []
        for r in runs:
            _, v, ok = cards[r][i]
            cells.append(f"{v} {'ok' if ok else 'XX'}")
        print(k.ljust(w) + "".join(f" | {c:>18s}" for c in cells))
    print("passed".ljust(w) + "".join(f" | {sum(ok for _, _, ok in cards[r]):>15d}/{len(cards[r])}" for r in runs))
