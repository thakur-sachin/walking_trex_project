"""Isaac Sim vs Isaac Lab comparison report from saved artifacts (no simulator needed).

    python3 walk_lab/compare_platforms.py --runs v8e v8e_lab v8e_lab_rsl > docs/PLATFORM_COMPARISON_RESULTS.md

Reads outputs/platform_bench.jsonl (walk_lab/bench_env.py), each run's train_log.jsonl and
walk_task.yaml, and eval_checkpoint.json / eval_checkpoint_lab.json (evaluate.py on the
Isaac Sim stack / on the Isaac Lab port) and prints markdown tables:
  1. simulator throughput per env count
  2. training: wall clock, samples/s, milestones (first iteration with a mean episode >= 15 s
     and >= 19 s under final conditions), final training metrics
  3. evaluation of every policy on BOTH simulators (survival, speed, heading, sway, push)
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "outputs" / "walk_runs"


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def fmt(v, nd=2):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def bench_table(tag: str | None) -> list[str]:
    rows = [r for r in load_jsonl(ROOT / "outputs" / "platform_bench.jsonl")
            if r.get("device", "").startswith("cuda") and (tag is None or r.get("tag") == tag)]
    if not rows:
        return ["(no GPU benchmark records yet: run walk_lab/bench_env.py on an idle GPU)"]
    latest = {}
    for r in rows:
        latest[(r["backend"], r["num_envs"])] = r
    out = ["| envs | Isaac Sim: ms/step | samples/s | GPU MiB | Isaac Lab: ms/step | samples/s | GPU MiB | Lab speedup |",
           "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in sorted({k[1] for k in latest}):
        s, l = latest.get(("isaacsim", n)), latest.get(("lab", n))
        sp = f"{s['ms_per_step'] / l['ms_per_step']:.2f}x" if s and l else "-"
        g = lambda r, k: fmt(r[k]) if r else "-"
        out.append(f"| {n} | {g(s, 'ms_per_step')} | {g(s, 'samples_per_s')} | {g(s, 'gpu_peak_mib')} | "
                   f"{g(l, 'ms_per_step')} | {g(l, 'samples_per_s')} | {g(l, 'gpu_peak_mib')} | {sp} |")
    st = {b: next((r for r in rows[::-1] if r["backend"] == b), None) for b in ("isaacsim", "lab")}
    out.append("")
    out.append("Startup: " + "; ".join(f"{b} app {r['app_start_s']} s + env build {r['env_build_s']} s ({r['num_envs']} envs)"
                                      for b, r in st.items() if r))
    return out


def train_summary(name: str) -> dict:
    log = load_jsonl(RUNS / name / "train_log.jsonl")
    if not log:
        return {}
    final = [r for r in log if r.get("harness_alpha", 1) == 0 and r.get("push_scale", 0) >= 1.0] or log
    ms = lambda thr: next((r for r in final if r.get("episode_length_s", 0) >= thr), None)
    tail = log[-50:]
    avg = lambda k: sum(r.get(k, 0.0) for r in tail) / len(tail)
    s15, s19 = ms(15.0), ms(19.0)
    return {
        "iters": log[-1]["iter"], "hours": log[-1].get("elapsed_min", 0) / 60.0,
        "s_per_it": sum(r.get("t_total", 0) for r in log) / len(log),
        # "samples" is cumulative in every train_log.jsonl (walk_rl PPO and the Lab env-side log)
        "samples_per_s": log[-1].get("samples", 0) / max(sum(r.get("t_total", 0) for r in log), 1e-9),
        "ep15": (s15["iter"], s15.get("elapsed_min", 0) / 60.0) if s15 else None,
        "ep19": (s19["iter"], s19.get("elapsed_min", 0) / 60.0) if s19 else None,
        "ep_len": avg("episode_length_s"), "term": avg("terminated_frac"), "rew": avg("mean_step_reward"),
        "dist": avg("distance_x_m"),
    }


def train_table(runs: list[str]) -> list[str]:
    out = ["| run | iterations | wall clock h | s/iter | samples/s | ep >= 15 s (it, h) | ep >= 19 s (it, h) | "
           "last-50 ep len s | term frac | step reward |",
           "|---|---:|---:|---:|---:|---|---|---:|---:|---:|"]
    for n in runs:
        s = train_summary(n)
        if not s:
            out.append(f"| {n} | (no train log) |||||||||")
            continue
        m = lambda v: f"{v[0]}, {v[1]:.2f}" if v else "not reached"
        out.append(f"| {n} | {s['iters']} | {s['hours']:.2f} | {s['s_per_it']:.2f} | {s['samples_per_s']:,.0f} | "
                   f"{m(s['ep15'])} | {m(s['ep19'])} | {s['ep_len']:.1f} | {s['term']:.2f} | {s['rew']:+.3f} |")
    out.append("")
    out.append("Step reward is comparable between runs on the same task config (walk_rl PPO vs rsl_rl both "
               "log the env's mean per-step reward). Training termination is not the fall rate; see evaluation.")
    return out


def eval_rows(name: str, suffix: str):
    p = RUNS / name / f"eval_checkpoint{suffix}.json"
    return json.loads(p.read_text()) if p.exists() else None


def eval_table(runs: list[str]) -> list[str]:
    out = ["| policy (trained on) | judged on | survival nominal | survival hard | speed %cmd @0.20 / 0.25 | "
           "heading deg/20 s (max) | roll p2p @0.25 | standing d_act | push walk 0.3 / 0.4 |",
           "|---|---|---|---|---|---:|---:|---:|---|"]
    for n in runs:
        meta = ""
        for suffix, sim in (("", "Isaac Sim"), ("_lab", "Isaac Lab")):
            e = eval_rows(n, suffix)
            if e is None:
                out.append(f"| {n} | {sim} | (not evaluated) |||||||")
                continue
            nom, hard = e["nominal"], e["hard"]
            by = lambda rows, c: next((r for r in rows if abs(r["cmd"] - c) < 1e-6), None)
            surv = lambda rows: f"{min(r['survive'] for r in rows) * 100:.0f}-{max(r['survive'] for r in rows) * 100:.0f}%"
            sp = lambda c: f"{by(nom, c)['vx'] / c * 100:.0f}" if by(nom, c) else "-"
            push = {(r["cmd"], r["kick_mps"]): r["survive"] for r in e.get("push", [])}
            pw = f"{push.get((0.1, 0.3), float('nan')):.2f} / {push.get((0.1, 0.4), float('nan')):.2f}"
            out.append(f"| {n}{meta} | {sim} | {surv(nom)} | {surv(hard)} | {sp(0.2)} / {sp(0.25)} | "
                       f"{max(r['|yaw|_deg'] for r in nom):.1f} | {fmt(by(nom, 0.25)['roll_p2p_deg'], 0) if by(nom, 0.25) else '-'} | "
                       f"{by(nom, 0.0)['d_action']:.3f} | {pw} |")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["v8e", "v8e_lab", "v8e_lab_rsl"])
    ap.add_argument("--bench-tag", default=None)
    a = ap.parse_args()
    print("## 1. Simulator throughput (same task, trained policy + noise, pushes on)\n")
    print("\n".join(bench_table(a.bench_tag)))
    print("\n## 2. Training\n")
    print("\n".join(train_table(a.runs)))
    print("\n## 3. Evaluation (evaluate.py, 256 envs, identical suite; each policy judged on both simulators)\n")
    print("\n".join(eval_table(a.runs)))
