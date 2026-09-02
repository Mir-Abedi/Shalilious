# Focused MNIST notebook experiments

`federated_local_sgd_kaggle.ipynb` reimplements four notebook-specific studies on
top of this repository's `src/` code. The notebook does not contain another client,
server, model, partitioner, drift definition, or FedAvg implementation.

## Shared protocol

- Dataset/model: normalized MNIST and `src.models.small_cnn` by default.
- Repetitions: seeds 0, 1, and 2. Conditions are paired by seed, including model
  initialization, partition, client minibatch streams, and participation sampling.
- Primary metric: the project's global training objective, evaluated over the actual
  mixture of client domains.
- Heterogeneity: `src.data.by_target_h`, reported using its realized
  `H_label = I(Client;Y)/H(Y)`. Quantity CV is reported separately.
- Drift: `src.server.Server.measure_drift`, namely the unweighted client mean
  `mean_i ||g_i-g||`, where every exact full-client gradient is evaluated at the same
  broadcast model and `g=sum_i (n_i/N)g_i`.
- Uncertainty: plots show the paired-seed mean and ±1 standard deviation. Raw checkpoint
  and summary CSVs are saved for every study.
- Budgets: learning curves are shown against both communication rounds and actual client
  mini-batch gradient updates. Diagnostic full-gradient passes are not counted as
  training work.

The default half-MNIST, three-seed configuration is intended to finish in roughly
90–120 minutes on a P100, but GPU load and notebook environment can change that estimate.
Set `DATA_FRAC=1.0` for a full-data final run or use the pilot settings in the configuration
cell before committing to a long run.

## Study 1 — unequal local work for a hard client

Two IID, near-equal shards have the same label mixture. Client 0 sees normalized MNIST;
client 1 sees a fixed Gaussian-corrupted view of its shard. `K_easy=5` is fixed while
`K_hard` is swept over `{1,3,5,8}`.

This asks whether extra local optimization closes the corrupted client's objective gap.
Because total work rises with `K_hard`, interpret both the domain-loss panel and the
reported computation count. The corruption is deterministic within each seed, so changing
`K_hard` does not also change the data. This is covariate shift, which `H_label` intentionally
does not claim to measure.

## Study 2 — number of clients

Full participation and `K=5` are held fixed while client count varies over `{2,5,10}`,
under both `H=0` and a common feasible `H=0.25`. These counts are used because the source
partitioner requires the number of MNIST clients to divide its ten label groups; 0.25 is
below the reachable ceiling for all three counts.

The communication-axis panel shows progress per synchronization. The computation-axis
panel exposes that a round with more clients performs more local updates. Realized
heterogeneity and quantity CV remain in the result table rather than being assumed from
requested values.

## Study 3 — communication-round weighting

The same seed-specific Dirichlet partition is reused across three endpoint-weighting rules:

- `uniform`: every online client has equal influence;
- `data`: standard FedAvg weights `n_i / sum_j n_j`;
- `inverse_js`: the notebook's exploratory rule
  `n_i exp(-beta JS(q_i,q))`, normalized over online clients.

The deliberately quantity-skewed partition makes uniform and data weighting genuinely
different. The global objective remains the data-size-weighted empirical objective for all
three methods, so uniform and inverse-JS may have a bias floor. A negative inverse-JS result
is therefore meaningful; it is not presented as a correction guaranteed to help.

`fednova` remains available in `src.studies` for unequal-`K_i` follow-ups, but it is omitted
from this equal-K sweep because it is algebraically identical to data-weighted FedAvg there.

## Study 4 — fraction of clients participating

With ten total clients and `K=5`, each round samples `ceil(qN)` clients without replacement
for `q in {0.2,0.5,1.0}`. The sweep is repeated at `H=0` and `H=0.8` using controlled source
partitions. Online-client data weights are renormalized; offline clients receive zero.

Round counts are scaled by `N/m`, so every condition receives the same total number of local
updates as the 30-round full-participation reference. This cleanly separates communication
cost from training computation. Requested and actual fractions, client coverage, and the CV
of participation counts are written to the summary. Availability is sampled anew by seed.

## Outputs

Each enabled study writes beneath `notebook_results/<study>/`:

- `history.csv`: checkpoint trajectories and budgets;
- `summary.csv`: one row per seed and condition;
- a PNG following the visual template used by the original project plots.

Tuple-valued audit fields such as selected client IDs, effective aggregation weights, and
per-client final metrics are JSON-encoded in CSV. The notebook also writes its resolved
configuration to `notebook_results/config.json`.

## Running

Open the notebook from the repository root, select a CUDA kernel, run setup/configuration,
then enable the desired `RUN_...` flags. Every study cell is independent and saves its output
before displaying the figure. Internet is needed only if MNIST is not already cached under
`DATA_ROOT`.

The identical studies can be launched without Jupyter:

```bash
jobs/run_notebook_experiments.sh --pilot
jobs/run_notebook_experiments.sh
```

The shell wrapper uses `.venv/bin/python` when present, otherwise `python3`, and requests
CUDA. Additional arguments are passed to `src/run_notebook_experiments.py`. Examples:

```bash
# Run only the hard-client and aggregation studies.
jobs/run_notebook_experiments.sh --studies hard_client aggregation

# Full MNIST, five paired seeds, custom output directory.
jobs/run_notebook_experiments.sh \
  --data-frac 1.0 --seeds 0 1 2 3 4 --output-root runs/notebook_extensions

# Show every CLI option.
python -u src/run_notebook_experiments.py --help
```
