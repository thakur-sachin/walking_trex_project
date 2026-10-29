#!/usr/bin/env bash
# TensorBoard for the walking runs.
#   walk_rl/start_tensorboard.sh [start|stop|status] [port]    (default: start, 6006)
#
# Runs two background processes and records their PIDs in outputs/tb/.pids:
#   1. walk_rl/tb_sync.py --watch  (jsonl training logs -> outputs/tb/<run>/ events;
#      picks up new iterations and brand-new runs automatically)
#   2. tensorboard --logdir outputs/tb
# `stop` only signals the PIDs recorded there (this machine is shared).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TB_DIR="$ROOT/outputs/tb"
PIDS="$TB_DIR/.pids"
CMD="${1:-start}"
PORT="${2:-6006}"
mkdir -p "$TB_DIR"

alive() { [ -f "$PIDS" ] && while read -r pid; do kill -0 "$pid" 2>/dev/null && return 0; done < "$PIDS"; return 1; }

case "$CMD" in
  start)
    if alive; then echo "already running (pids: $(tr '\n' ' ' < "$PIDS")); http://localhost:$PORT"; exit 0; fi
    if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
      echo "port $PORT is already in use by another process; pick one: $0 start <port>" >&2; exit 1
    fi
    nohup /usr/bin/python3 "$ROOT/walk_rl/tb_sync.py" --watch > "$TB_DIR/sync.log" 2>&1 &
    echo $! > "$PIDS"
    sleep 3  # let the first sync create the event dirs
    nohup "${TENSORBOARD:-$HOME/.local/bin/tensorboard}" --logdir "$TB_DIR" --port "$PORT" --reload_interval 5 \
      > "$TB_DIR/tensorboard.log" 2>&1 &
    echo $! >> "$PIDS"
    echo "TensorBoard: http://localhost:$PORT   (logs: $TB_DIR/*.log)"
    ;;
  stop)
    if [ -f "$PIDS" ]; then
      while read -r pid; do kill "$pid" 2>/dev/null && echo "stopped $pid" || true; done < "$PIDS"
      rm -f "$PIDS"
    else echo "nothing recorded to stop"; fi
    ;;
  status)
    if alive; then echo "running (pids: $(tr '\n' ' ' < "$PIDS")); http://localhost:$PORT"; else echo "not running"; fi
    ;;
  *) echo "usage: $0 [start|stop|status] [port]" >&2; exit 2 ;;
esac
