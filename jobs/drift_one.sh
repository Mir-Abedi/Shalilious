#!/usr/bin/env bash
# All three seeds at one H_label -- one condor job per H.
# Usage: drift_one.sh <H_label>      Writes 3 CSVs into runs/.
set -euo pipefail
cd "$(dirname "$0")/.."

H="${1:?target H_label in [0,1]}"

PY=.venv/bin/python
[ -x "$PY" ] || PY=python3   # the cluster image provides its own interpreter

for SEED in 0 1 2; do
  echo "=== H=$H seed=$SEED ==="
  "$PY" src/main.py \
    --partition target_h --target-h "$H" --seed "$SEED" \
    --clients 20 --rounds 100 --local-steps 5 \
    --model small_cnn --lr 0.05 --batch-size 64 \
    --eval-every 10 --drift-every 5
done
