#!/usr/bin/env bash
# Train a list of experiments one after another on the GPU, and run each one's checks (run_experiment.sh
# SKIP_TRAIN=1) in the background while the next one trains. One training saturates the RTX 4060; the checks
# (Isaac Sim PhysX evaluation, MuJoCo) fit beside it.
#   walk_rl/queue_experiments.sh QUEUE_FILE
# QUEUE_FILE lines: NAME TASK_YAML [KEY=VALUE ...] [-- COMPARE_RUN ...]   (# comments; KEY=VALUE are env for both
# steps, e.g. ITERATIONS=3000 SEED=1 TRAIN_BACKEND=lab3). A line is read when its turn comes, so the file can grow.
set -uo pipefail
Q=$1
cd "$(dirname "$0")/.."
LOGS=${LOGS:-outputs/queue_logs}; mkdir -p "$LOGS"
n=0
while true; do
  n=$((n + 1))
  line=$(grep -v '^\s*#' "$Q" | grep -v '^\s*$' | sed -n "${n}p")
  [ -z "$line" ] && break
  read -r -a parts <<< "$line"
  name=${parts[0]}; task=${parts[1]}; envs=(); cmp=()
  rest=("${parts[@]:2}"); seen=0
  for x in "${rest[@]}"; do
    if [ "$x" = "--" ]; then seen=1; elif [ $seen = 1 ]; then cmp+=("$x"); else envs+=("$x"); fi
  done
  echo "[$(date -u +%H:%MZ)] queue: train $name ${envs[*]}"
  env "${envs[@]}" TRAIN_ONLY=1 walk_rl/run_experiment.sh "$name" "$task" > "$LOGS/$name.train.log" 2>&1
  if grep -q "train FAILED" "$LOGS/$name.train.log"; then echo "[$(date -u +%H:%MZ)] queue: $name train FAILED"; continue; fi
  echo "[$(date -u +%H:%MZ)] queue: checks $name (background)"
  ( env "${envs[@]}" SKIP_TRAIN=1 walk_rl/run_experiment.sh "$name" "$task" "${cmp[@]}" > "$LOGS/$name.checks.log" 2>&1
    echo "[$(date -u +%H:%MZ)] queue: $name checks done" ) &
done
wait
echo "[$(date -u +%H:%MZ)] queue: all done"
