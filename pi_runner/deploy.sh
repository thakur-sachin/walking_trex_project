#!/usr/bin/env bash
# Copy what the Pi runs (pi_runner, configs, chosen policies) and set up its venv.
#   pi_runner/deploy.sh [user@host] [run ...]      default: pi@trex.local v9a
# On the Pi: cd ~/trex && .venv/bin/python -m pi_runner.hw_check info
set -euo pipefail
HOST=${1:-pi@trex.local}
shift || true
RUNS=("${@:-v9a}")
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST=trex
cd "$ROOT"
files=(pi_runner/__init__.py pi_runner/runner.py pi_runner/policy.py pi_runner/bno055.py pi_runner/lx16a.py
       pi_runner/hardware.py pi_runner/hw_check.py pi_runner/requirements-pi.txt pi_runner/pi_setup.sh pi_runner/servo_identify.py pi_runner/servo_limits.py pi_runner/servo_hold.py
       common/__init__.py common/robot_config.py walk_rl/config.py walk_rl/walk_task.yaml
       assets/robots/trex/robot_config.yaml assets/robots/trex/hardware.yaml)
for r in "${RUNS[@]}"; do
  files+=("outputs/walk_runs/$r/policy_numpy.npz" "outputs/walk_runs/$r/walk_task.yaml")
done
ssh "$HOST" "mkdir -p ~/$DEST/walk_rl && touch ~/$DEST/walk_rl/__init__.py"
rsync -aR --chmod=F644,Fu+x "${files[@]}" "$HOST:$DEST/"
ssh "$HOST" "cd ~/$DEST && { [ -x .venv/bin/python ] || python3 -m venv .venv; } \
  && .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -q -r pi_runner/requirements-pi.txt \
  && echo '$(git rev-parse --short HEAD) $(git branch --show-current)' > DEPLOYED_FROM \
  && .venv/bin/python -c 'import numpy, yaml, serial, smbus2; print(\"venv ok: numpy\", numpy.__version__)'"
