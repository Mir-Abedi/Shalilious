#!/usr/bin/env bash
# Centralized SGD reference for the sync-interval sweep -- the ceiling on every panel.
#
# Local SGD with ONE client IS centralized SGD: aggregation over a single client is the
# identity and no drift is possible. So this is plain batch-64 SGD over the same 5e6
# sample-gradient budget, logged every 1000 steps. No special code path needed.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
[ -x "$PY" ] || PY=python3

mkdir -p runs/sync
for SEED in 0 1 2; do
  echo "=== centralized SGD seed=$SEED ==="
  "$PY" src/main.py \
    --partition iid --seed "$SEED" \
    --clients 1 --rounds 78 --local-steps 1000 \
    --model small_cnn --lr 0.05 --batch-size 64 \
    --eval-every 1 --drift-every 0 \
    --out "runs/sync/central_s${SEED}.csv"
done
