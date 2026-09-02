#!/usr/bin/env bash
# One cell of the trust-region sweep: both heterogeneity levels x 3 seeds at one (rho, K).
# Usage: ball_one.sh <rho|inf> <local_steps K>
#
# Equal gradient budget of 100 epochs, matching experiment 2, so the rho=inf column is
# directly comparable to the plain Local SGD results there.
set -euo pipefail
cd "$(dirname "$0")/.."

RHO="${1:?rho, or inf for no ball}"
K="${2:?local steps}"

PY=.venv/bin/python
[ -x "$PY" ] || PY=python3

CLIENTS=20
BATCH=64
BUDGET=5000000
ROUNDS=$("$PY" -c "print(max(1, round($BUDGET/($CLIENTS*$K*$BATCH))))")
EVERY=$("$PY" -c "print(max(1, $ROUNDS//100))")

mkdir -p runs/ball
echo "rho=$RHO K=$K -> $ROUNDS rounds, eval every $EVERY"

for H in 1.0 0.75; do
  for SEED in 0 1 2; do
    echo "=== rho=$RHO K=$K H=$H seed=$SEED ==="
    "$PY" src/main.py \
      --partition target_h --target-h "$H" --seed "$SEED" \
      --clients $CLIENTS --rounds "$ROUNDS" --local-steps "$K" \
      --model small_cnn --lr 0.05 --batch-size $BATCH \
      --rho "$RHO" --grad-batch 2048 \
      --eval-every "$EVERY" --exact-cos-every 10 --drift-every 0 \
      --out "runs/ball/K${K}_rho${RHO}_h${H}_s${SEED}.csv"
  done
done
