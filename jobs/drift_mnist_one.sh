#!/usr/bin/env bash
# All three seeds at one H_label on MNIST -- one condor job per H.
# Usage: [MODEL=small_cnn|linear] [CLIENTS=10|5|2] jobs/drift_mnist_one.sh <H_label>
# Writes 3 CSVs into runs/mnist/drift_${MODEL}_n${CLIENTS}/.
#
# The MNIST translation of jobs/drift_one.sh; see jobs/experiments_MNIST.md.
set -euo pipefail
cd "$(dirname "$0")/.."

H="${1:?target H_label in [0,1]}"
MODEL="${MODEL:-small_cnn}"

PY=.venv/bin/python
[ -x "$PY" ] || PY=python3   # the cluster image provides its own interpreter

# target_h needs n_clients to divide the label groups it skews over, and here those are the
# 10 digits, so the only choices are 10, 5 and 2. 10 is the one that spans the full H_label
# range [0, 1], but it also gives each client a single digit at the top of that range, where
# the local objective degenerates -- see the H=1.0 turnover in jobs/experiments_MNIST.md.
# CLIENTS=5 (two digits each, ceiling H=0.699) is the control for that.
CLIENTS="${CLIENTS:-10}"

# Rounds are set to spend 768000 sample-gradients = 12.8 epochs of MNIST's 60000, whatever
# the client count -- the same epochs the CIFAR-100 sweep spent (100 x 20 x 5 x 64 / 50000).
# A round is clients x K x batch samples, so halving the clients doubles the rounds.
ROUNDS=$(( 768000 / (CLIENTS * 5 * 64) ))
DRIFT_EVERY=$(( ROUNDS / 24 ))     # 24 drift measurements per run, at any client count
EVAL_EVERY=$(( ROUNDS / 12 ))

OUT="runs/mnist/drift_${MODEL}_n${CLIENTS}"
mkdir -p "$OUT"

for SEED in 0 1 2; do
  echo "=== MNIST $MODEL n=$CLIENTS H=$H seed=$SEED ($ROUNDS rounds) ==="
  "$PY" src/main.py --dataset mnist \
    --partition target_h --target-h "$H" --seed "$SEED" \
    --clients "$CLIENTS" --rounds $ROUNDS --local-steps 5 \
    --model "$MODEL" --lr 0.05 --batch-size 64 \
    --eval-every $EVAL_EVERY --drift-every $DRIFT_EVERY \
    --out "$OUT/targeth${H}_s${SEED}.csv"
done
