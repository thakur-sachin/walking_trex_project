#!/usr/bin/env bash
# Isaac Sim vs Isaac Lab comparison steps (same task, same config, same seed).
#
#   walk_lab/run_comparison.sh bench                 # throughput, both stacks, 1024..16384 envs (idle GPU!)
#   walk_lab/run_comparison.sh eval <run>            # evaluate.py on BOTH simulators -> eval_checkpoint{,_lab}.json
#   walk_lab/run_comparison.sh train-lab <run> <its> <task.yaml>   # walk_rl PPO on the Isaac Lab env
#   walk_lab/run_comparison.sh train-rsl <run> <its> <task.yaml>   # rsl_rl on the Isaac Lab env
#   walk_lab/run_comparison.sh report                # docs/PLATFORM_COMPARISON_RESULTS.md
#
# Isaac Sim stack: conda env isaac6. Isaac Lab port: conda env isaaclab6.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMNI_KIT_ACCEPT_EULA=YES
SIM_PY=${SIM_PY:-$HOME/miniconda3/envs/isaac6/bin/python}
LAB_PY=${LAB_PY:-$HOME/miniconda3/envs/isaaclab6/bin/python}
LOGS=outputs/logs
mkdir -p "$LOGS"

case "${1:-}" in
  bench)
    for n in 1024 4096 8192 16384; do
      for b in isaacsim lab; do
        py=$SIM_PY; [ "$b" = lab ] && py=$LAB_PY
        echo "[cmp] bench $b $n envs"
        $py walk_lab/bench_env.py --backend "$b" --num-envs "$n" --tag gpu_idle > "$LOGS/bench_${b}_${n}.log" 2>&1 || true
        grep '^\[bench\]' "$LOGS/bench_${b}_${n}.log" || echo "[cmp] bench $b $n FAILED (see $LOGS/bench_${b}_${n}.log)"
      done
    done ;;
  eval)
    run=$2
    echo "[cmp] eval $run on Isaac Sim"
    $SIM_PY walk_rl/evaluate.py --name "$run" > "outputs/walk_runs/$run/eval_checkpoint.log" 2>&1
    echo "[cmp] eval $run on Isaac Lab"
    $LAB_PY walk_rl/evaluate.py --backend lab --name "$run" > "outputs/walk_runs/$run/eval_checkpoint_lab.log" 2>&1
    grep -c '^\[eval\]' "outputs/walk_runs/$run/eval_checkpoint.log" "outputs/walk_runs/$run/eval_checkpoint_lab.log" ;;
  train-lab)
    $LAB_PY walk_rl/train.py --backend lab --name "$2" --iterations "$3" --task "$4" > "$LOGS/train_$2.log" 2>&1 ;;
  train-rsl)
    $LAB_PY walk_lab/train_rsl.py --name "$2" --iterations "$3" --task "$4" > "$LOGS/train_$2.log" 2>&1
    # rsl_rl's own TensorBoard events next to the tb_sync ones
    ln -sfn "$(pwd)/outputs/walk_runs/$2/rsl_rl" "outputs/tb/$2_rsl_native" ;;
  report)
    python3 walk_lab/compare_platforms.py --runs v8e v8e_lab v8e_lab_rsl --bench-tag gpu_idle > docs/PLATFORM_COMPARISON_RESULTS.md
    echo "[cmp] wrote docs/PLATFORM_COMPARISON_RESULTS.md" ;;
  *) sed -n 2,10p "$0"; exit 1 ;;
esac
