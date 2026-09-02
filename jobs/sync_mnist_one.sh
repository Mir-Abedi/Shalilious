#!/usr/bin/env bash
# One cell of the MNIST sync-interval sweep: all three seeds at one (K, H).
# Usage: sync_mnist_one.sh <local_steps K> <H_label>
#
# FIXED COMMUNICATION BUDGET, unlike the CIFAR-100 version. Every cell runs exactly 100
# rounds, so the gradient budget grows with K (64000*K sample-gradients: ~1 epoch at K=1,
# 160 at K=150). The question is therefore "given 100 rounds of communication, how much
# local work should each client do", not "what does syncing less cost at equal compute".
# The per-panel ceiling in jobs/sgd_mnist_matched.sh is what makes the panels comparable.
set -euo pipefail
cd "$(dirname "$0")/.."

K="${1:?local steps}"
H="${2:?target H_label in [0,1]}"

PY=.venv/bin/python
[ -x "$PY" ] || PY=python3

CLIENTS=10     # must divide the 10 MNIST digits for target_h
BATCH=64
ROUNDS=100
EVERY=2        # 50 points per curve

mkdir -p runs/mnist/sync
echo "K=$K H=$H -> $ROUNDS rounds, $((ROUNDS * CLIENTS * K * BATCH)) sample-gradients"

for SEED in 0 1 2; do
  echo "=== MNIST K=$K H=$H seed=$SEED ==="
  "$PY" src/main.py --dataset mnist \
    --partition target_h --target-h "$H" --seed "$SEED" \
    --clients $CLIENTS --rounds $ROUNDS --local-steps "$K" \
    --model small_cnn --lr 0.05 --batch-size $BATCH \
    --eval-every $EVERY --drift-every 0 \
    --out "runs/mnist/sync/K${K}_h${H}_s${SEED}.csv"
done
