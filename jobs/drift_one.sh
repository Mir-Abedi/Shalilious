#!/usr/bin/env bash
# One run of the drift-vs-heterogeneity sweep.  Usage: drift_one.sh <H_label> <seed>
# H_label is the heterogeneity to GENERATE, in [0, 1].  Writes runs/<generated name>.csv .
set -euo pipefail
cd "$(dirname "$0")/.."

H="${1:?target H_label in [0,1]}"
SEED="${2:?seed}"

PY=.venv/bin/python
[ -x "$PY" ] || PY=python3   # the cluster may provide its own interpreter

exec "$PY" src/main.py \
  --partition target_h --target-h "$H" --seed "$SEED" \
  --clients 20 --rounds 100 --local-steps 5 \
  --model small_cnn --lr 0.05 --batch-size 64 \
  --eval-every 10 --drift-every 5
