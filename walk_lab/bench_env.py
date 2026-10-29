"""Simulator throughput benchmark, identical for both stacks (same task, same actions).

    python walk_lab/bench_env.py --backend isaacsim --num-envs 4096          # env isaac6
    python walk_lab/bench_env.py --backend lab      --num-envs 4096          # env isaaclab6

Measures env construction time, then steps a trained policy plus exploration noise with
training-time settings (pushes on, harness off) and reports control steps/s, samples/s
(env steps x num_envs), ms per control step (8 physics substeps), and peak GPU memory of
this process (nvidia-smi). Appends one JSON line to outputs/platform_bench.jsonl.
Run one configuration per process (each stack owns a single simulation context) and only
on an otherwise idle GPU: the numbers are meaningless next to a training run.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=["isaacsim", "lab"], required=True)
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--steps", type=int, default=240, help="timed control steps (10 PPO rollouts of 24)")
parser.add_argument("--warmup", type=int, default=48)
parser.add_argument("--run", default="v8d", help="policy that drives the robots (so they walk, fall and reset like in training)")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--tag", default="")
args = parser.parse_args()

from walk_rl import backend  # noqa: E402

t_app = time.time()
simulation_app = backend.launch(args.backend, device=args.device)
t_app = time.time() - t_app

import torch  # noqa: E402

peak = {"mib": 0}


def _watch_gpu(stop):
    pid = str(os.getpid())
    while not stop.is_set():
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout
            for line in out.splitlines():
                p, m = [s.strip() for s in line.split(",")]
                if p == pid:
                    peak["mib"] = max(peak["mib"], int(m))
        except Exception:
            pass
        stop.wait(0.5)


try:
    from walk_rl.ppo import ActorCritic
    from walk_rl.task_core import apply_checkpoint_interface, load_task_config

    stop = threading.Event()
    threading.Thread(target=_watch_gpu, args=(stop,), daemon=True).start()
    run_dir = PROJECT_ROOT / "outputs" / "walk_runs" / args.run
    cfg = load_task_config(run_dir / "walk_task.yaml")
    ck = torch.load(run_dir / "checkpoint.pt", map_location="cpu")
    apply_checkpoint_interface(cfg, ck["meta"])
    EnvCls = backend.env_class(args.backend)
    t0 = time.time()
    env = EnvCls(args.num_envs, cfg, device=args.device, simulation_app=simulation_app, seed=0)
    t_build = time.time() - t0
    ac = ActorCritic(env.num_actor_obs, env.num_critic_obs, env.num_actions, cfg.ppo.actor_hidden,
                     cfg.ppo.critic_hidden, cfg.ppo.init_log_std).to(env.device)
    ac.load_state_dict(ck["model"])
    env.set_push_scale(1.0)
    obs = env.reset()
    sync = (lambda: torch.cuda.synchronize()) if "cuda" in str(env.device) else (lambda: None)

    def run(n):
        global obs
        for _ in range(n):
            with torch.no_grad():
                a = ac.act_deterministic(obs["actor"]) + 0.25 * torch.randn(env.num_envs, env.num_actions, device=env.device)
            obs, _, _, _ = env.step(a)

    run(args.warmup)
    sync()
    t0 = time.time()
    run(args.steps)
    sync()
    el = time.time() - t0
    stop.set()
    rec = {"backend": args.backend, "num_envs": args.num_envs, "device": args.device, "steps": args.steps,
           "app_start_s": round(t_app, 1), "env_build_s": round(t_build, 1),
           "ms_per_step": round(1000 * el / args.steps, 2), "steps_per_s": round(args.steps / el, 1),
           "samples_per_s": round(args.steps * args.num_envs / el), "gpu_peak_mib": peak["mib"],
           "torch_peak_alloc_mib": round(torch.cuda.max_memory_allocated() / 2**20) if "cuda" in str(env.device) else 0,
           "dr": env.dr_active, "tag": args.tag, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    print("[bench] " + json.dumps(rec), flush=True)
    with open(PROJECT_ROOT / "outputs" / "platform_bench.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
except Exception:
    import traceback

    traceback.print_exc()
    sys.stdout.flush()
finally:
    simulation_app.close()
