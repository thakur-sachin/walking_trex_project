"""MuJoCo validation of every policy on the robot model it was trained on (its "home" model).

    $LAB pi_runner/sim2sim_home_model.py [--runs v9m v9o ...] [--jobs 8]

The simulated robot changed seven times (models M0-M7, docs/ABLATION_STUDY.md), so the standard sim2sim
numbers of an old policy on today's robot say little about the policy. This script rebuilds each run's own
robot from the robot_config.yaml it trained with and runs the standard sweep on it:
  frictions 0.3 0.5 0.8 x delays 0 1 2 x commands 0 0.05 0.10 0.20, 60 s each (36 episodes),
  with the servo model the run trained with (actuator.model in the run's walk_task.yaml).

robot_config source, first found:
  1. outputs/walk_runs/<run>/robot_config.yaml (copied at training time since 2026-10-21)
  2. the run's config commit in walk_rl/experiments/registry.yaml (git show <commit>:assets/.../robot_config.yaml)
  3. the first commit of the run's robot model (MODEL_COMMIT)
The URDF has not changed since 2026-10-09, so robot_config is the whole model. Limit: MuJoCo always routes
the ankle torque through the linkage; M0 (Isaac teleported ankle, unrouted) cannot be reproduced exactly.

The numbers say how well a policy transfers to a second physics engine on its own robot. They do not rank
policies across models: an easier robot model can score higher.
Writes outputs/walk_runs/<run>/sim2sim_home.json (+ robot_config_home.yaml) and outputs/sim2sim_home_model.json.
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS = PROJECT_ROOT / "outputs" / "walk_runs"
RC_PATH = "assets/robots/trex/robot_config.yaml"
# first commit of each robot model's robot_config (docs/ABLATION_STUDY.md model table)
MODEL_COMMIT = {"M0": "1d31a0e", "M1": "0047918", "M2": "0059ed7", "M3": "8c2e1cb", "M4": "af2091b",
                "M5": "f20acf4", "M6": "5fba0ee",
                "M7": "6bb7d71"}
SWEEP = ["--frictions", "0.3", "0.5", "0.8", "--delays", "0", "1", "2", "--commands", "0.0", "0.05", "0.10", "0.20"]


def home_config(run: str, entry: dict) -> tuple[Path, str]:
    out = RUNS / run / "robot_config_home.yaml"
    own = RUNS / run / "robot_config.yaml"
    if own.exists():
        out.write_text(own.read_text())
        return out, "run copy"
    commit = (entry.get("config") or "").split(":")[0] or MODEL_COMMIT[entry["model"]]
    src = "registry commit" if entry.get("config") else f"{entry['model']} first commit"
    text = subprocess.run(["git", "show", f"{commit}:{RC_PATH}"], cwd=PROJECT_ROOT, capture_output=True, text=True,
                          check=True).stdout
    out.write_text(text)
    return out, f"{src} {commit}"


def summarize(path: Path) -> dict:
    rows = json.loads(path.read_text())["rows"]
    walk = [r for r in rows if r["cmd"] > 0]
    pct = lambda c: 100 * sum(r["vx"] for r in rows if r["cmd"] == c) / sum(r["cmd"] == c for r in rows) / c  # noqa: E731
    fwd = sum(abs(r["forward_m"]) for r in walk)
    return {"survived": sum(r["fell"] is None for r in rows), "episodes": len(rows),
            "speed_pct": {c: round(pct(c)) for c in (0.05, 0.10, 0.20)},
            "side_per_forward_pct": round(100 * sum(abs(r["side_m"]) for r in walk) / max(fwd, 1e-6)),
            "yaw_abs_deg": round(sum(abs(r["yaw_deg"]) for r in walk) / len(walk), 1)}


def run_one(run: str, entry: dict, python: str) -> dict:
    cfg, src = home_config(run, entry)
    task = yaml.safe_load((RUNS / run / "walk_task.yaml").read_text()) or {}
    actuator = (task.get("actuator") or {}).get("model", "implicit_pd")
    out = RUNS / run / "sim2sim_home.json"
    cmd = [python, "pi_runner/sim2sim.py", "--run", run, *SWEEP, "--actuator", actuator,
           "--robot-config", str(cfg), "--out", str(out)]
    log = RUNS / run / "sim2sim_home.log"
    with log.open("w") as f:
        rc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    res = {"run": run, "model": entry["model"], "actuator": actuator, "robot_config": src}
    if rc != 0 or not out.exists():
        res["error"] = log.read_text().strip().splitlines()[-1][:200]
    else:
        res |= summarize(out)
    print(f"[home] {run}: " + json.dumps({k: v for k, v in res.items() if k != 'run'}), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", help="default: every registry run with a policy_numpy.npz")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    reg = yaml.safe_load((PROJECT_ROOT / "walk_rl" / "experiments" / "registry.yaml").read_text())
    entries = {e["id"]: e for e in (reg["runs"] if isinstance(reg, dict) else reg)}
    runs = a.runs or [r for r in entries if (RUNS / r / "policy_numpy.npz").exists()]
    with ThreadPoolExecutor(a.jobs) as ex:
        results = list(ex.map(lambda r: run_one(r, entries[r], sys.executable), runs))
    (PROJECT_ROOT / "outputs" / "sim2sim_home_model.json").write_text(json.dumps(results, indent=1))
    print("\n| run | model | servo | survived | speed @0.05 / 0.10 / 0.20 | side / forward | mean abs yaw | robot_config |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        if "error" in r:
            print(f"| {r['run']} | {r['model']} | {r['actuator']} | error: {r['error']} | | | | {r['robot_config']} |")
            continue
        s = r["speed_pct"]
        print(f"| {r['run']} | {r['model']} | {r['actuator']} | {r['survived']}/{r['episodes']} | "
              f"{s[0.05]} / {s[0.10]} / {s[0.20]} % | {r['side_per_forward_pct']} % | {r['yaw_abs_deg']} deg | {r['robot_config']} |")


if __name__ == "__main__":
    main()
