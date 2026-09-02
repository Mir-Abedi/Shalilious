#!/usr/bin/env bash
# One cell of the MNIST sync-interval sweep: all three seeds at one (K, H).
# Usage: sync_mnist_one.sh <local_steps K> <H_label>
#
# Equal gradient budget, exactly as the CIFAR-100 version: every cell spends 10 epochs =
# 6e5 sample-gradients, so rounds shrink as K grows and the only thing varying is how
# often clients synchronize. Any loss difference across panels is the cost of syncing
# less, not a difference in compute.
set -euo pipefail
cd "$(dirname "$0")/.."

K="${1:?local steps}"
H="${2:?target H_label in [0,1]}"

PY=.venv/bin/python
[ -x "$PY" ] || PY=python3

CLIENTS=10                          # must divide the 10 MNIST digits for target_h
BATCH=64
BUDGET=600000                       # 10 epochs x 60000 samples
ROUNDS=$("$PY" -c "print(max(1, round($BUDGET/($CLIENTS*$K*$BATCH))))")
EVERY=$("$PY" -c "print(max(1, $ROUNDS//100))")   # ~100 points per curve

mkdir -p runs/mnist/sync
echo "K=$K H=$H -> $ROUNDS rounds, eval every $EVERY"

for SEED in 0 1 2; do
  echo "=== MNIST K=$K H=$H seed=$SEED ==="
  "$PY" src/main.py --dataset mnist \
    --partition target_h --target-h "$H" --seed "$SEED" \
    --clients $CLIENTS --rounds "$ROUNDS" --local-steps "$K" \
    --model small_cnn --lr 0.05 --batch-size $BATCH \
    --eval-every "$EVERY" --drift-every 0 \
    --out "runs/mnist/sync/K${K}_h${H}_s${SEED}.csv"
done
