#!/usr/bin/env bash
# Centralized SGD at the distributed arm's EFFECTIVE batch: clients x client batch
# = 10 x 64 = 640. One job, three seeds.
#
# The like-for-like reference for the whole grid: one local step plus shard-size-weighted
# averaging IS batch-640 SGD, so this curve must land on top of the K=1 curves, and the
# distance from it at larger K is the cost of synchronizing less. Batch 640 rather than
# 64 on purpose -- CIFAR-100 experiment 2 established that a batch-64 reference measures a
# batch-size effect, not a cost of distribution.
#
# 938 updates x 640 = 600320 sample-gradients, the same 10-epoch budget every cell spends.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
[ -x "$PY" ] || PY=python3

mkdir -p runs/mnist/sync
for SEED in 0 1 2; do
  echo "=== MNIST centralized SGD, batch 640, seed=$SEED ==="
  "$PY" src/main.py --dataset mnist \
    --partition iid --seed "$SEED" \
    --clients 1 --rounds 938 --local-steps 1 \
    --model small_cnn --lr 0.05 --batch-size 640 \
    --eval-every 9 --drift-every 0 \
    --out "runs/mnist/sync/matched_s${SEED}.csv"
done
