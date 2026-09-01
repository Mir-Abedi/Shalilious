#!/usr/bin/env bash
# Smallest run that proves both arms train. See docs/superpowers/specs/.
set -euo pipefail
cd "$(dirname "$0")/.."
for K in 1 5; do
  .venv/bin/python src/main.py --model small_cnn --data-frac 0.1 --clients 4 \
    --rounds 100 --eval-every 5 --local-steps "$K"
done
