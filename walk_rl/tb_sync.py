"""Mirror walking-run logs into TensorBoard event files.

Training writes one JSON line per PPO iteration to
outputs/walk_runs/<run>/train_log.jsonl. This script converts those into
TensorBoard events under outputs/tb/<run>/ and (with --watch) keeps tailing, so
new iterations -- and new runs -- show up live without touching the trainer or
the isaac6 env. It only needs the `tensorboard` package (system python3).

    python3 walk_rl/tb_sync.py            # one-shot rebuild of every run
    python3 walk_rl/tb_sync.py --watch    # keep syncing (what start_tensorboard.sh runs)

Tags: reward/*, episode/*, reward_terms/*, ppo/*, perf/*; the exact task config
used by each run is attached as text ("config") so runs are comparable.
"""
import argparse
import json
import math
import shutil
import time
from pathlib import Path

from tensorboard.compat.proto import event_pb2, summary_pb2, tensor_pb2, tensor_shape_pb2, types_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "outputs" / "walk_runs"
TB_DIR = PROJECT_ROOT / "outputs" / "tb"

# (jsonl key) -> (tensorboard tag)
SCALARS = {
    "mean_step_reward": "reward/mean_step",
    "episode_length_s": "episode/length_s",
    "distance_x_m": "episode/distance_x_m",
    "yaw_drift_deg": "episode/yaw_drift_deg",
    "terminated_frac": "episode/terminated_frac",
    "mean_cmd_vx": "episode/mean_cmd_vx",
    "episodes": "episode/episodes_finished",
    "kl": "ppo/kl",
    "lr": "ppo/lr",
    "entropy": "ppo/entropy",
    "loss_pi": "ppo/loss_policy",
    "loss_v": "ppo/loss_value",
    "action_std": "ppo/action_std",
    "epochs_run": "ppo/epochs_run",
    "t_collect": "perf/collect_s",
    "t_total": "perf/iteration_s",
    "harness_alpha": "curriculum/harness_alpha",
    "samples": "perf/samples",
    "elapsed_min": "perf/elapsed_min",
}


def _scalar_event(tag: str, value: float, step: int, wall: float) -> event_pb2.Event:
    val = summary_pb2.Summary.Value(tag=tag, simple_value=float(value))
    return event_pb2.Event(wall_time=wall, step=step, summary=summary_pb2.Summary(value=[val]))


def _text_event(tag: str, text: str, wall: float) -> event_pb2.Event:
    meta = summary_pb2.SummaryMetadata(plugin_data=summary_pb2.SummaryMetadata.PluginData(plugin_name="text"))
    tensor = tensor_pb2.TensorProto(
        dtype=types_pb2.DT_STRING,
        string_val=[text.encode()],
        tensor_shape=tensor_shape_pb2.TensorShapeProto(dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1)]),
    )
    val = summary_pb2.Summary.Value(tag=tag, metadata=meta, tensor=tensor)
    return event_pb2.Event(wall_time=wall, step=0, summary=summary_pb2.Summary(value=[val]))


class RunSync:
    """Incrementally converts one run's jsonl into an event file."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.out_dir = TB_DIR / run_dir.name
        if self.out_dir.exists():
            shutil.rmtree(self.out_dir)  # rebuild on start so restarts never duplicate points
        self.out_dir.mkdir(parents=True)
        self.writer = EventFileWriter(str(self.out_dir), max_queue_size=10, flush_secs=2)
        self.offset = 0
        self.t0 = (run_dir / "walk_task.yaml").stat().st_mtime if (run_dir / "walk_task.yaml").exists() else time.time()
        cfg = run_dir / "walk_task.yaml"
        if cfg.exists():
            self.writer.add_event(_text_event("config/walk_task.yaml", "```yaml\n" + cfg.read_text() + "\n```", self.t0))
        self.last_iter = -1

    def poll(self) -> int:
        log = self.run_dir / "train_log.jsonl"
        if not log.exists():
            return 0
        n = 0
        with open(log, "rb") as f:
            f.seek(self.offset)
            for raw in f:
                if not raw.endswith(b"\n"):  # partially written line; pick it up next poll
                    break
                self.offset += len(raw)
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                step = int(rec["iter"])
                wall = self.t0 + 60.0 * float(rec.get("elapsed_min", 0.0))
                for key, tag in SCALARS.items():
                    v = rec.get(key)
                    if isinstance(v, (int, float)) and math.isfinite(v):
                        self.writer.add_event(_scalar_event(tag, v, step, wall))
                for key, v in rec.items():
                    if key.startswith("term/") and isinstance(v, (int, float)) and math.isfinite(v):
                        self.writer.add_event(_scalar_event("reward_terms/" + key[5:], v, step, wall))
                self.last_iter = step
                n += 1
        if n:
            self.writer.flush()
        return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()

    syncs: dict[str, RunSync] = {}
    while True:
        if RUNS_DIR.exists():
            for d in sorted(RUNS_DIR.iterdir()):
                if d.is_dir() and (d / "train_log.jsonl").exists() and d.name not in syncs:
                    syncs[d.name] = RunSync(d)
                    print(f"[tb_sync] tracking run '{d.name}'", flush=True)
        for name, rs in syncs.items():
            n = rs.poll()
            if n and args.watch:
                print(f"[tb_sync] {name}: +{n} iterations (now at {rs.last_iter})", flush=True)
        if not args.watch:
            for name, rs in syncs.items():
                print(f"[tb_sync] {name}: synced through iteration {rs.last_iter}")
                rs.writer.close()
            print(f"[tb_sync] event files in {TB_DIR}")
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
