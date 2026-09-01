#!/usr/bin/env bash
# Full drift-vs-heterogeneity sweep, run sequentially.  See jobs/experiments.md.
# 11 heterogeneity settings x 3 seeds = 33 runs.  On the cluster use drift_sweep.sub instead.
set -euo pipefail
cd "$(dirname "$0")/.."

ALPHAS=(1000 10 3 1 0.5 0.3 0.1 0.05 0.01)
SEEDS=(0 1 2)

for s in "${SEEDS[@]}"; do
  jobs/drift_one.sh iid 0 "$s"              # H_label ~ 0 anchor
  jobs/drift_one.sh by_superclass 0 "$s"    # H_label == 1 anchor
  for a in "${ALPHAS[@]}"; do
    jobs/drift_one.sh dirichlet "$a" "$s"
  done
done
