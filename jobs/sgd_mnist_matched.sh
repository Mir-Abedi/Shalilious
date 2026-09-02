#!/usr/bin/env bash
# The per-panel centralized ceiling for the MNIST sync sweep.
# Usage: sgd_mnist_matched.sh <local_steps K>
#
# Centralized SGD given the SAME sample budget as panel K, at the distributed arm's
# effective batch (10 clients x 64 = 640): 100*K updates of batch 640 is 64000*K
# sample-gradients, exactly what the K panel spends. It answers "what could this compute
# have bought with no synchronization at all", which is the ceiling each panel is missing
# once the panels no longer share a budget.
#
# At K=1 it is not merely a reference but an identity check: one local step plus
# shard-size-weighted averaging IS batch-640 SGD, so this must land on the K=1 curves.
set -euo pipefail
cd "$(dirname "$0")/.."

K="${1:?local steps of the panel being matched}"

PY=.venv/bin/python
[ -x "$PY" ] || PY=python3

mkdir -p runs/mnist/sync
for SEED in 0 1 2; do
  echo "=== MNIST centralized SGD, batch 640, matching K=$K, seed=$SEED ==="
  "$PY" src/main.py --dataset mnist \
    --partition iid --seed "$SEED" \
    --clients 1 --rounds 100 --local-steps "$K" \
    --model small_cnn --lr 0.05 --batch-size 640 \
    --eval-every 2 --drift-every 0 \
    --out "runs/mnist/sync/centralK${K}_s${SEED}.csv"
done
