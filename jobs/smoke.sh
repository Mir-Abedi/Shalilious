#!/usr/bin/env bash
# Smallest run that proves both arms train. See docs/superpowers/specs/.
set -euo pipefail
cd "$(dirname "$0")/.."
for K in 1 5; do
  .venv/bin/python src/main.py --model small_cnn --data-frac 0.1 --clients 4 \
    --rounds 100 --eval-every 5 --local-steps "$K"
done

# MNIST arm: both models, and target_h over the 10 digits (clients must divide 10).
for MODEL in linear small_cnn; do
  .venv/bin/python src/main.py --dataset mnist --model "$MODEL" --data-frac 0.1 \
    --clients 5 --partition target_h --target-h 0.5 \
    --rounds 50 --eval-every 5 --local-steps 5
done
