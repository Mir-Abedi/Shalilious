#!/usr/bin/env bash
# Full drift-vs-heterogeneity sweep, run sequentially.  See jobs/experiments.md.
# 10 heterogeneity settings x 3 seeds = 30 runs.  On the cluster use drift_sweep.sub instead.
set -euo pipefail
cd "$(dirname "$0")/.."

ALPHAS=(1000 10 3 1 0.5 0.3 0.1 0.05 0.01)
SEEDS=(0 1 2)

for s in "${SEEDS[@]}"; do
  jobs/drift_one.sh iid 0 "$s"            # the alpha -> infinity anchor
  for a in "${ALPHAS[@]}"; do
    jobs/drift_one.sh dirichlet "$a" "$s"
  done
done
