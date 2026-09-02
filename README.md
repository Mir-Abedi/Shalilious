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

## Focused MNIST experiments

The notebook and terminal runner add four focused experiments. They use the
same implementations in `src/`; the notebook is only an interactive front end.
The default full profile uses 50% of MNIST, the `small_cnn`, 30 reference rounds,
batch size 64, learning rate 0.05, and seeds 0, 1, and 2. This profile is sized
for approximately 90–120 minutes on a P100. The `--pilot` profile uses one seed,
10% of MNIST, and four reference rounds.

### Shared implementation and analysis

Every condition within a seed starts from the same seeded global-model
initialization. Each client also has a private seeded minibatch generator, so
changing the number of clients or rerunning another method does not consume a
shared random stream in an uncontrolled order. Partitions, corruption, and
participation sampling are regenerated from recorded seeds. The intent is to
measure variation across seeds without allowing methods to receive unrelated
initial conditions.

The primary metric is the training loss of the actual distributed objective. If
client $i$ owns $n_i$ examples and has loss $F_i$, the reported objective is

$$
F(w)=\sum_i \frac{n_i}{\sum_j n_j}F_i(w).
$$

For the hard-client experiment this includes the corrupted client domain; it is
not evaluated on an easier clean substitute. Training accuracy is saved as a
secondary diagnostic, while loss remains the main optimization measure.

Progress is recorded against two budgets:

- `round`: completed server communication rounds;
- `local_updates`: client minibatch gradient updates actually performed.

`examples_processed` is also saved, since the final minibatch or a very small
client shard may contain fewer than 64 examples. Plotting both rounds and local
updates prevents a method from appearing communication-efficient merely because
it spends substantially more computation during every round.

The drift measurement is the definition from `src/server.py`, not endpoint
parameter distance:

$$
g=\sum_i\frac{n_i}{N}g_i,\qquad
D(w)=\frac{1}{M}\sum_i\lVert g_i-g\rVert_2.
$$

Here, $N=\sum_i n_i$, $M$ is the number of clients, and each $g_i$ is the exact
full-shard gradient evaluated at the same global
broadcast model. Therefore the spread is attributable to client objectives, not
to clients being evaluated at different local iterates. Exact drift is measured
at initialization and at the final model because each measurement requires a
full backward pass over all client data. These diagnostic passes are not counted
as training work.

Controlled label heterogeneity uses `src/data.py::by_target_h`. The value shown
in tables is always the realized

$$
H_{label}=I(Client;Y)/H(Y),
$$

rather than only the requested target. `quantity_cv` is reported separately so
label skew and unequal client sizes are not silently conflated. Curves show the
mean across paired seeds with a shaded or error-bar band of ±1 standard
deviation. Raw seed-level results remain available in CSV; no best-seed selection
is performed.

### Experiment 1: unequal local steps for a hard client

This experiment asks whether allocating more local computation to a difficult
client improves its domain or instead increases drift.

- Two clients receive one fixed IID split with near-equal sample counts and
  nearly the same label distribution.
- Client 0 sees ordinary normalized MNIST and always uses `K_easy=5`.
- Client 1 sees deterministic Gaussian corruption with standard deviation 0.8
  after normalization. Its local steps are swept over
  `K_hard in {1, 3, 5, 8}`.
- The corruption is fixed by example index and seed, and the same two shards are
  reused for every `K_hard` value in that seed.
- Standard data-size-weighted FedAvg is used at communication.

The first plot panel compares the final clean-client and corrupted-client losses.
The second reports final exact gradient drift. The summary also contains
per-client accuracy, global loss, local-update count, and processed examples.
When interpreting the sweep, note that `K_easy` is fixed, so a larger `K_hard`
also increases total computation. A lower hard-domain loss is evidence that the
extra allocation helped that domain, but it is not by itself evidence of greater
compute efficiency. The local-update totals provide that necessary context.

This experiment studies covariate shift. Its difficulty is deliberately not
described by `H_label`, because label histograms can remain almost identical even
when image quality differs substantially.

### Experiment 2: number of clients

This experiment asks what changes when the same dataset is divided among more
federated clients.

- Client count is swept over `{2, 5, 10}`.
- Every client participates in every round and uses `K=5`.
- The model, learning rate, batch size, and 30-round budget are fixed.
- Each client count is run at controlled `H_label=0` and at a shared feasible
  target `H_label=0.25`.

The values 2, 5, and 10 are intentional: `by_target_h` requires the number of
MNIST clients to divide the ten digit groups. A target of 0.25 is reachable even
for two clients, so all client counts can be compared at the same requested
heterogeneity instead of quietly receiving different skew levels.

For each heterogeneity condition, the left plot compares loss by communication
round. The right plot shows the same trajectories by local updates. With full
participation, increasing the client count increases work per round, so the two
panels answer different questions:

- the round axis measures communication efficiency;
- the update axis reveals whether an apparent round-level improvement survives
  after accounting for extra computation.

The analysis should verify realized `h_label` and `quantity_cv` in `summary.csv`
before attributing differences solely to federation size.

### Experiment 3: weighted aggregation at communication

This experiment asks how client endpoint weights affect optimization when client
sample counts and label distributions differ.

- Ten clients use one seed-specific Dirichlet partition with `alpha=0.3`.
- The identical partition, initialization, and client data streams are reused
  across aggregation policies within each seed.
- Every client participates and uses `K=5`, so local computation is controlled.
- Three policies are compared:

  - `uniform`: each client endpoint gets weight $1/M$;
  - `data`: standard FedAvg weight $n_i/\sum_jn_j$;
  - `inverse_js`: exploratory weight
    $n_i\exp[-\beta JS(q_i,q)]$, renormalized with `beta=2`.

The global evaluation objective stays data-size weighted for all policies.
Consequently, uniform and inverse-JS aggregation are not granted a different
evaluation target that favors their own weights. They may have a nonzero bias
floor, and a negative inverse-JS result is scientifically meaningful rather than
treated as a failed implementation.

Plots compare loss against both rounds and local updates. Every logged round also
records selected client IDs and the normalized effective aggregation weights,
allowing the communication calculation to be audited from `history.csv`.
`fednova` is implemented for unequal-$K_i$ follow-ups, but it is omitted from
this default equal-K sweep because it
reduces algebraically to data-weighted FedAvg when every client uses the same K.

### Experiment 4: fraction of clients participating

This experiment asks how intermittent availability changes communication and
computation efficiency, particularly under label skew.

- The federation always contains ten clients with `K=5`.
- Participation fractions are `q in {0.2, 0.5, 1.0}`.
- Each round samples `m=ceil(qN)` clients without replacement.
- The study is repeated at controlled `H_label=0` and `H_label=0.8`.
- Selected clients use data-size weights renormalized over the online set;
  offline clients receive weight zero for that round.

To make computation comparable, the number of rounds is scaled by `N/m`. With a
30-round full-participation reference, `q=1.0` runs 30 rounds, `q=0.5` runs 60,
and `q=0.2` runs 150. All three therefore end at the same planned number of local
updates. The round-axis panels expose the extra communications required by low
participation, while the update-axis panels compare learning at matched training
computation.

Participation is resampled by seed. The summary reports the requested and actual
fraction, clients per round, fraction of clients ever selected, and coefficient
of variation of client selection counts. Under `H_label=0.8`, a small online set
can omit large portions of the label distribution in a round; the seed variation
and participation statistics are therefore part of the result, not incidental
logging.

### Generated results

The README describes the implemented analysis protocol; it does not invent a
numerical conclusion before these configurations have been run. Empirical claims
should be made from the generated seed-level CSVs and uncertainty plots.

Each study directory contains:

- `history.csv`: every communication round, selected client IDs, effective
  weights, training budgets, checkpoint losses, and drift checkpoints;
- `summary.csv`: final seed-level metrics and realized partition/participation
  statistics for every condition;
- one PNG with the mean and ±1 standard-deviation analysis;
- `config.json` at the output root with the complete resolved invocation.

Notebook execution writes to `notebook_results/`. The terminal runner writes to
`notebook_results_terminal/` unless `--output-root` is supplied.

The code implementing these studies is split as follows:

- `src/studies.py` — shared study server, aggregation, participation, metrics,
  and single-condition runner;
- `src/run_notebook_experiments.py` — terminal orchestration and CLI;
- `federated_local_sgd_kaggle.ipynb` — interactive orchestration;
- `plots/notebook_experiments.py` — seed aggregation and plot templates;
- `jobs/run_notebook_experiments.sh` — CUDA server wrapper.

## Layout

- `jobs/` — runner scripts (`.sh`, `.sub`)
- `plots/` — generated figures
- `federated_local_sgd_kaggle.ipynb` — focused MNIST extensions built on `src/`
- `jobs/experiments_<DATASET>.md` — what was run, and where its results and plots live

## Running

`.venv/bin/python src/main.py --local-steps 1` is mini-batch SGD; `--local-steps 5` is Local SGD.
Add `--partition dirichlet --alpha 0.1` for a skewed split, `--dataset mnist` to switch
datasets. Results land in `runs/<dataset>/*.csv`.
Self-checks: `.venv/bin/python src/test_data.py`.

Run the four notebook-extension studies on a GPU with
`jobs/run_notebook_experiments.sh --pilot`, then `jobs/run_notebook_experiments.sh`.
