#!/usr/bin/env bash
# Train one experiment and run the fixed evaluation on it (about 2 h on an RTX 4060 laptop).
#   walk_rl/run_experiment.sh NAME TASK_YAML [COMPARE_RUN ...]
#   walk_rl/run_experiment.sh v9o walk_rl/experiments/v9o_mirror_loss.yaml v9m v9n
# Steps: train (1500 iterations), Pi-runner parity, Isaac evaluation on the lx16a servo model, MuJoCo
# sim2sim sweep, MuJoCo video, gait scorecard against COMPARE_RUN..., ablation study. Logs go to
# outputs/walk_runs/NAME/. Add the run to walk_rl/experiments/registry.yaml before you start.
# TRAIN_BACKEND=lab3 trains on Isaac Lab 3 + Newton (walk_lab3); every check after training stays the same
# (Isaac Sim PhysX evaluation, MuJoCo sim2sim), so a Newton-trained policy is judged on two other engines.
# TRAIN_ONLY=1 stops after training; SKIP_TRAIN=1 runs only the checks (walk_rl/queue_experiments.sh overlaps
# one run's checks with the next run's training). ITERATIONS (default 1500), SEED (default 0).
set -uo pipefail
NAME=$1; TASK=$2; shift 2
cd "$(dirname "$0")/.."
export OMNI_KIT_ACCEPT_EULA=YES
PY=${PY:-~/miniconda3/envs/isaac6/bin/python}; LAB=${LAB:-~/miniconda3/envs/isaaclab6/bin/python}
NEWTON=${NEWTON:-~/trex_stack/IsaacLab_dev/.venv/bin/python}; TRAIN_BACKEND=${TRAIN_BACKEND:-isaacsim}
if [ "$TRAIN_BACKEND" = lab3 ]; then TRAIN_PY=$NEWTON; else TRAIN_PY=$PY; fi
OUT=outputs/walk_runs/$NAME
step(){ echo "[$(date -u +%H:%MZ)] $1"; }
mkdir -p "$OUT"
if [ -z "${SKIP_TRAIN:-}" ]; then
step "train $NAME"
$TRAIN_PY walk_rl/train.py --backend "$TRAIN_BACKEND" --name "$NAME" --iterations "${ITERATIONS:-1500}" --seed "${SEED:-0}" --task "$TASK" > "$OUT/train.log" 2>&1 || { echo "train FAILED (see $OUT/train.log)"; exit 1; }
fi
[ -n "${TRAIN_ONLY:-}" ] && { step "trained"; exit 0; }
step "runner parity"
$PY pi_runner/test_obs_parity.py --run "$NAME" > "$OUT/parity.log" 2>&1
grep "\[parity\] \(OK\|FAILED\)" "$OUT/parity.log" || echo "parity did not finish (see $OUT/parity.log)"
step "Isaac evaluation"
$PY walk_rl/evaluate.py --name "$NAME" --actuator lx16a --commands 0.0 0.05 0.075 0.10 0.15 0.20 0.25 > "$OUT/eval_checkpoint_lx16a.log" 2>&1 || echo "evaluation FAILED"
step "MuJoCo sim2sim"
$LAB pi_runner/sim2sim.py --run "$NAME" --frictions 0.3 0.5 0.8 --commands 0.0 0.05 0.10 0.20 --actuator lx16a > "$OUT/sim2sim_lx16a.log" 2>&1 || echo "sim2sim FAILED"
step "MuJoCo sim2sim, low friction"
$LAB pi_runner/sim2sim.py --run "$NAME" --frictions 0.15 0.2 0.27 --commands 0.0 0.05 0.10 0.20 --actuator lx16a --out "$OUT/sim2sim_lowmu_lx16a.json" > "$OUT/sim2sim_lowmu.log" 2>&1 || echo "sim2sim low friction FAILED"
step "MuJoCo sim2sim, real-like (audit 2026-10-27)"
P=1 LAB=$LAB lab3/rescore_reallike.sh "$NAME"
python3 lab3/rank_reallike.py "$@" "$NAME"
step "MuJoCo video"
$LAB pi_runner/render_mujoco.py --run "$NAME" --schedule 0:2 0.05:6 0.10:6 0.15:6 0.20:6 0:3 --tag speeds > "$OUT/render.log" 2>&1 || echo "render FAILED"
step "gait scorecard"
python3 walk_rl/gait_scorecard.py "$@" "$NAME"
step "ablation study"
python3 walk_rl/ablation_study.py
step "done"
