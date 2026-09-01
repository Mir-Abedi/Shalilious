# Local SGD under Heterogeneous Data

Simulation study of distributed optimization with a small number of clients,
comparing **mini-batch SGD** against **Local SGD / FedAvg**.

## Dataset

**CIFAR-100** — trained on the 100 fine labels, partitioned by the 20 coarse
superclasses.

## What we investigate

- IID vs. a prescribed non-IID client partition.
- Optimization progress measured two ways: per **gradient computation** and per
  **communication round**.
- How data heterogeneity changes the benefit (or cost) of taking several local
  steps between communication rounds.

## Independent investigation

Sweeps over the number of local steps, degree of heterogeneity, number of
clients, and the fraction of participating clients per round.

## Layout

- `jobs/` — runner scripts (`.sh`, `.sub`)
- `plots/` — generated figures

## Running

`.venv/bin/python src/main.py --local-steps 1` is mini-batch SGD; `--local-steps 5` is Local SGD.
Add `--partition dirichlet --alpha 0.1` for a skewed split. Results land in `runs/*.csv`.
Self-checks: `.venv/bin/python src/test_data.py`.
