"""Running ablation study: the registry of runs joined with their measured results.

    python walk_rl/ablation_study.py            # writes docs/ABLATION_STUDY.md and outputs/ablation_study.json

walk_rl/experiments/registry.yaml holds what each run changed, why, its targets, verdict and lesson (hand
written). This script adds what the run directories measured (outputs/walk_runs/<id>/):
- training: iterations, minutes, final action std (train_log.jsonl);
- Isaac: speed at 0.10 m/s as % of the command, worst walking survival nominal / hard sensors, worst servo
  saturation (eval_checkpoint_lx16a.json, else eval_checkpoint.json on the ideal servo);
- MuJoCo: episodes that survive, median speed at 0.10 (sim2sim_policy_numpy_lx16a.json, else the ideal-servo file);
- MuJoCo on the run's own robot model: the same sweep rebuilt from the robot_config it trained with
  (sim2sim_home.json, pi_runner/sim2sim_home_model.py);
- the gait scorecard pass count where sim2sim has the gait statistics.
Each result is labelled with the robot model it was measured on: from its provenance record when it has one,
else from the file time against the model start times in the registry. A number measured on another model
than the run trained on is marked, because numbers are only comparable within one model.
"""
import json
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "walk_rl"))

import gait_scorecard  # noqa: E402

RUNS = PROJECT_ROOT / "outputs" / "walk_runs"
REGISTRY = PROJECT_ROOT / "walk_rl" / "experiments" / "registry.yaml"


def model_at(models: dict, t: datetime) -> str:
    cur = None
    for k, m in models.items():
        if datetime.fromisoformat(m["from"]) <= t:
            cur = k
    return cur


def first(run: str, names: list[str]) -> Path | None:
    for n in names:
        p = RUNS / run / n
        if p.exists():
            return p
    return None


def measured_model(models: dict, path: Path, data: dict) -> str:
    """Model of a result file: provenance hash is exact for new files; the file time is used for old ones."""
    t = datetime.fromisoformat(data["provenance"]["time"]) if "provenance" in data else datetime.fromtimestamp(path.stat().st_mtime)
    return model_at(models, t)


def measure(run: str, models: dict) -> dict:
    out = {}
    log = RUNS / run / "train_log.jsonl"
    if log.exists():
        lines = log.read_text().splitlines()
        last = json.loads(lines[-1])
        out["train"] = {"iterations": last["iter"], "minutes": round(last.get("elapsed_min", 0)),
                        "envs": last.get("samples", 0) // max(last["iter"], 1) // 24, "final_std": round(last.get("action_std", 0), 2)}
    # judge a run on the servo model it trained with; the other servo is a transfer test, not its result
    cfg_p = RUNS / run / "walk_task.yaml"
    cfg = yaml.safe_load(cfg_p.read_text()) if cfg_p.exists() else {}
    lx = ((cfg or {}).get("actuator") or {}).get("model") == "lx16a"
    out["servo"] = "lx16a" if lx else "ideal"
    ev_p = first(run, ["eval_checkpoint_lx16a.json"] if lx else ["eval_checkpoint.json", "eval_best.json"])
    if ev_p:
        ev = json.loads(ev_p.read_text())
        walk = [r for r in ev["nominal"] if r["cmd"] > 0]
        at10 = [r for r in ev["nominal"] if abs(r["cmd"] - 0.10) < 1e-6]
        out["isaac"] = {
            "servo": "lx16a" if "lx16a" in ev_p.name else "ideal",
            "model": measured_model(models, ev_p, ev),
            "speed_010": round(100 * at10[0]["vx"] / 0.10) if at10 else None,
            "surv_min": round(min(r["survive"] for r in walk), 2) if walk else None,
            "hard_min": round(min(r["survive"] for r in ev.get("hard", []) if r["cmd"] > 0), 2) if ev.get("hard") else None,
            "sat_max": round(100 * max(r["sat"] for r in walk), 1) if walk and "sat" in walk[0] else None,
        }
    s2_p = first(run, ["sim2sim_policy_numpy_lx16a.json"] if lx else ["sim2sim_policy_numpy.json"])
    if s2_p:
        s2 = json.loads(s2_p.read_text())
        rows = s2["rows"]
        r10 = [r for r in rows if abs(r["cmd"] - 0.10) < 1e-6]
        out["mujoco"] = {
            "servo": "lx16a" if "lx16a" in s2_p.name else "ideal",
            "model": measured_model(models, s2_p, s2),
            "survived": f"{sum(r['fell'] is None for r in rows)}/{len(rows)}",
            "speed_010": round(100 * st.median(r["vx"] for r in r10) / 0.10) if r10 else None,
        }
    home_p = RUNS / run / "sim2sim_home.json"  # pi_runner/sim2sim_home_model.py: same sweep on the run's own robot model
    if home_p.exists():
        rows = json.loads(home_p.read_text())["rows"]
        r10 = [r for r in rows if abs(r["cmd"] - 0.10) < 1e-6]
        out["mujoco_home"] = {"survived": f"{sum(r['fell'] is None for r in rows)}/{len(rows)}",
                              "speed_010": round(100 * st.median(r["vx"] for r in r10) / 0.10) if r10 else None}
    if ev_p and ev_p.name == "eval_checkpoint_lx16a.json" and s2_p and s2_p.name.endswith("_lx16a.json"):
        try:
            card = gait_scorecard.score(run)
            if not any(k == "gait statistics" for k, _, _ in card):
                out["scorecard"] = {"passed": sum(ok for _, _, ok in card), "of": len(card),
                                    "fails": [k for k, _, ok in card if not ok]}
        except (KeyError, IndexError, ZeroDivisionError):
            pass
    return out


def cell(v, suffix: str = "") -> str:
    return "—" if v is None else f"{v}{suffix}"


def main():
    reg = yaml.safe_load(REGISTRY.read_text())
    models = reg["models"]
    runs = []
    for r in reg["runs"]:
        m = measure(r["id"], models)
        for k, v in (r.get("measured_on") or {}).items():
            if k in m:
                m[k]["model"] = v
        runs.append({**r, "measured": m})

    L = ["# Ablation study", "",
         "Every training run of the Trex walking task: what it changed, why, what was measured and what was learned.",
         f"Generated by `walk_rl/ablation_study.py` from `walk_rl/experiments/registry.yaml` and `outputs/walk_runs/` "
         f"({datetime.now():%Y-%m-%d %H:%M}). Do not edit by hand.", "",
         "**How to read it.** Each run makes one change against its parent. Results are only comparable within one robot "
         "model: the simulator's robot changed six times as measurements came in. A value marked † was measured on a "
         "later model than the run trained on (the file was re-run after a model change). Each run is judged on the servo "
         "model it trained with: `ideal` is the PD drive of M0/M1, `lx16a` the fitted LX-16A model. Isaac numbers are walking at 0.10 m/s with randomization on; MuJoCo numbers "
         "are the deployed NumPy loop over friction × bus delay (median speed). \"MuJoCo own model\" is the same sweep with the robot rebuilt from the "
         "robot_config the run trained with, so every run is scored on its own robot; it shows transfer to a second engine, "
         "not a ranking across models. Old run configs: `git show <config>`.", "",
         "## Robot models", "", "| model | from | change | detail |", "|---|---|---|---|"]
    for k, m in models.items():
        L.append(f"| {k} | {m['from'].replace('T', ' ')} | {m['name']} | {m['note']} |")
    L += ["", "## Runs", "",
          "| run | parent | model | servo | change | Isaac speed @0.10 | Isaac hard surv. | MuJoCo survived | MuJoCo speed @0.10 | MuJoCo own model: survived | own model: speed @0.10 | scorecard | real robot | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        m = r["measured"]
        iz, mj, sc, mh = m.get("isaac", {}), m.get("mujoco", {}), m.get("scorecard"), m.get("mujoco_home", {})
        dag = lambda d: "†" if d and d.get("model") and d["model"] != r["model"] else ""  # noqa: E731
        L.append(f"| **{r['id']}** | {r.get('parent', '')} | {r['model']} | {m.get('servo', '—')} | {r['change']} | "
                 f"{cell(iz.get('speed_010'), '%')}{dag(iz)} | {cell(iz.get('hard_min'))}{dag(iz)} | "
                 f"{cell(mj.get('survived'))}{dag(mj)} | {cell(mj.get('speed_010'), '%')}{dag(mj)} | "
                 f"{cell(mh.get('survived'))} | {cell(mh.get('speed_010'), '%')} | "
                 f"{cell(sc and str(sc['passed']) + '/' + str(sc['of']))} | {r.get('real', '—')} | {r['verdict']} |")
    L += ["", "## Run details", ""]
    for r in runs:
        m = r["measured"]
        L.append(f"### {r['id']}" + (f" ← {r['parent']}" if r.get("parent") else "") + f" · {r['model']} · {r['verdict']}")
        L.append("")
        for k, lab in [("change", "Change"), ("why", "Why"), ("target", "Targets"), ("result", "Result"),
                       ("real", "Real robot"), ("lesson", "Lesson"), ("stack", "Stack")]:
            if r.get(k):
                L.append(f"- **{lab}:** {r[k]}")
        if "train" in m:
            t = m["train"]
            L.append(f"- **Training:** {t['iterations']} iterations, {t['envs']} envs, {t['minutes']} min, final action std {t['final_std']}")
        if m.get("scorecard"):
            sc = m["scorecard"]
            L.append(f"- **Scorecard:** {sc['passed']}/{sc['of']}" + (f"; fails: {', '.join(sc['fails'])}" if sc["fails"] else ""))
        if r.get("config"):
            L.append(f"- **Config:** `git show {r['config']}`")
        L.append("")
    (PROJECT_ROOT / "docs" / "ABLATION_STUDY.md").write_text("\n".join(L))
    out = PROJECT_ROOT / "outputs" / "ablation_study.json"
    out.write_text(json.dumps({"generated": f"{datetime.now():%Y-%m-%dT%H:%M}", "models": models, "runs": runs}, indent=1))
    print(f"[ablation] {len(runs)} runs -> docs/ABLATION_STUDY.md, {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
