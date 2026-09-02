# Local SGD under Heterogeneous Data

Simulation study of distributed optimization with a small number of clients,
comparing **mini-batch SGD** against **Local SGD / FedAvg**.

## Datasets

- **CIFAR-100** (default) — trained on the 100 fine labels, partitioned by the
  20 coarse superclasses.
- **MNIST** (`--dataset mnist`) — 10 digits, which are both the targets and the
  axis the client skew is drawn over. Cheap enough for the convex arm.

Models: `small_cnn` (both datasets), `resnet18`, and `linear` — multinomial
logistic regression, a convex objective.

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
- `jobs/experiments_<DATASET>.md` — what was run, and where its results and plots live

## Running

`.venv/bin/python src/main.py --local-steps 1` is mini-batch SGD; `--local-steps 5` is Local SGD.
Add `--partition dirichlet --alpha 0.1` for a skewed split, `--dataset mnist` to switch
datasets. Results land in `runs/<dataset>/*.csv`.
Self-checks: `.venv/bin/python src/test_data.py`.
