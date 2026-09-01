#!/usr/bin/env bash
# One run of the drift-vs-heterogeneity sweep.  Usage: drift_one.sh <partition> <alpha> <seed>
# Writes runs/<generated name>.csv .  Called directly by drift_sweep.sh and by condor.
set -euo pipefail
cd "$(dirname "$0")/.."

PARTITION="${1:?partition (iid|dirichlet)}"
ALPHA="${2:?alpha (ignored for iid, pass 0)}"
SEED="${3:?seed}"

PY=.venv/bin/python
[ -x "$PY" ] || PY=python3   # the cluster may provide its own interpreter

exec "$PY" src/main.py \
  --partition "$PARTITION" --alpha "$ALPHA" --seed "$SEED" \
  --clients 10 --rounds 100 --local-steps 5 \
  --model small_cnn --lr 0.05 --batch-size 64 \
  --eval-every 10 --drift-every 5
