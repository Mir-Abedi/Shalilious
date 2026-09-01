#!/usr/bin/env bash
# Full drift-vs-heterogeneity sweep, run sequentially.  See jobs/experiments.md.
# 21 H values, 3 seeds each.  On the cluster use drift_sweep.sub instead (21 parallel jobs).
set -euo pipefail
cd "$(dirname "$0")/.."

for h in 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0; do
  jobs/drift_one.sh "$h"
done
