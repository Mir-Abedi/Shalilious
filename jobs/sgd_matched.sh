#!/usr/bin/env bash
# Centralized SGD at the distributed arm's EFFECTIVE batch size (20 clients x 64 = 1280).
#
# This is the like-for-like control for K=1: one local step plus weighted averaging IS
# batch-1280 SGD, so this curve should land on top of the K=1 curve. The batch-64
# reference in sgd_baseline.sh answers a different question -- what plain small-batch SGD
# achieves on the same SAMPLE budget, where it gets 20x more parameter updates.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
[ -x "$PY" ] || PY=python3

mkdir -p runs/sync
for SEED in 0 1 2; do
  echo "=== centralized SGD, batch 1280, seed=$SEED ==="
  "$PY" src/main.py \
    --partition iid --seed "$SEED" \
    --clients 1 --rounds 100 --local-steps 39 \
    --model small_cnn --lr 0.05 --batch-size 1280 \
    --eval-every 1 --drift-every 0 \
    --out "runs/sync/matched_s${SEED}.csv"
done
