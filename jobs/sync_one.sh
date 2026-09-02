#!/usr/bin/env bash
# One cell of the sync-interval sweep: all three seeds at one (K, H).
# Usage: sync_one.sh <local_steps K> <H_label>
#
# Equal gradient budget: every cell spends 100 epochs = 5e6 sample-gradients,
# so rounds shrink as K grows and the only thing varying is how often clients sync.
set -euo pipefail
cd "$(dirname "$0")/.."

K="${1:?local steps}"
H="${2:?target H_label in [0,1]}"

PY=.venv/bin/python
[ -x "$PY" ] || PY=python3

CLIENTS=20
BATCH=64
BUDGET=5000000                      # 100 epochs x 50000 samples
ROUNDS=$("$PY" -c "print(max(1, round($BUDGET/($CLIENTS*$K*$BATCH))))")
EVERY=$("$PY" -c "print(max(1, $ROUNDS//100))")   # ~100 points per curve

mkdir -p runs/sync
echo "K=$K H=$H -> $ROUNDS rounds, eval every $EVERY"

for SEED in 0 1 2; do
  echo "=== K=$K H=$H seed=$SEED ==="
  "$PY" src/main.py \
    --partition target_h --target-h "$H" --seed "$SEED" \
    --clients $CLIENTS --rounds "$ROUNDS" --local-steps "$K" \
    --model small_cnn --lr 0.05 --batch-size $BATCH \
    --eval-every "$EVERY" --drift-every 0 \
    --out "runs/sync/K${K}_h${H}_s${SEED}.csv"
done
