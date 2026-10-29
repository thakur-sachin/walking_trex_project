#!/usr/bin/env bash
# Re-score runs in the "real-like" MuJoCo grid (audit 2026-10-27): the fitted LX-16A with the drive-torque
# limit and supply sag, supply 6.0 / 6.7 V, sensor noise and encoder offsets, IMU 1-2 control steps late,
# bus delay 0-1, friction 0.2 / 0.27 / 0.4 (wood 0.27), commands 0.05 / 0.10, 30 s.
#   lab3/rescore_reallike.sh RUN [RUN ...]      (P=6 parallel jobs; POLICY=policy_numpy.npz)
# Writes outputs/walk_runs/<run>/sim2sim_reallike.json (sim2sim_reallike_<policy stem>.json for another policy).
cd "$(dirname "$0")/.."
LAB=${LAB:-~/miniconda3/envs/isaaclab6/bin/python}
POLICY=${POLICY:-policy_numpy.npz}
stem=$(basename "$POLICY" .npz); [ "$stem" = policy_numpy ] && suf="" || suf="_$stem"
printf '%s\n' "$@" | xargs -P "${P:-6}" -I{} sh -c "$LAB pi_runner/sim2sim.py --run {} --policy $POLICY --actuator lx16a \
  --seconds 30 --frictions 0.2 0.27 0.4 --delays 0 1 --bus-voltage 6.0 6.7 --imu-delay 1 2 --commands 0.05 0.10 \
  --noisy --supply-sag --drive-limit --out outputs/walk_runs/{}/sim2sim_reallike$suf.json > outputs/walk_runs/{}/sim2sim_reallike$suf.log 2>&1 \
  && echo '[rescore] {} done' || echo '[rescore] {} FAILED'"
