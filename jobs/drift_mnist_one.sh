#!/usr/bin/env bash
# All three seeds at one H_label on MNIST -- one condor job per H.
# Usage: [MODEL=small_cnn|linear] jobs/drift_mnist_one.sh <H_label>
# Writes 3 CSVs into runs/mnist/drift_$MODEL/.
#
# The MNIST translation of jobs/drift_one.sh; see jobs/experiments_MNIST.md.
set -euo pipefail
cd "$(dirname "$0")/.."

H="${1:?target H_label in [0,1]}"
MODEL="${MODEL:-small_cnn}"

PY=.venv/bin/python
[ -x "$PY" ] || PY=python3   # the cluster image provides its own interpreter

# 10 clients, not the 20 used on CIFAR-100: target_h needs n_clients to divide the label
# groups it skews over, and here those are the 10 digits. 10 is also the only client count
# that spans the full H_label range [0, 1].
CLIENTS=10

# 240 rounds x 10 clients x 5 steps x 64 = 768000 sample-gradients = 12.8 epochs of MNIST's
# 60000, matching the epochs the CIFAR-100 sweep spent (100 x 20 x 5 x 64 / 50000).
ROUNDS=240

OUT="runs/mnist/drift_$MODEL"
mkdir -p "$OUT"

for SEED in 0 1 2; do
  echo "=== MNIST $MODEL H=$H seed=$SEED ==="
  "$PY" src/main.py --dataset mnist \
    --partition target_h --target-h "$H" --seed "$SEED" \
    --clients $CLIENTS --rounds $ROUNDS --local-steps 5 \
    --model "$MODEL" --lr 0.05 --batch-size 64 \
    --eval-every 20 --drift-every 10 \
    --out "$OUT/targeth${H}_s${SEED}.csv"
done
