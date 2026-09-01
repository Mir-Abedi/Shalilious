# Experiments

## 1. Client drift vs. label heterogeneity

**Claim.** Client drift grows with data heterogeneity.

**x-axis — heterogeneity.** The normalized mutual information between client identity
`C` and label `Y`, measured on the partition that was actually drawn (not the nominal
alpha):

    H_label = I(C;Y) / H(Y) = 1 - H(Y|C) / H(Y)      in [0, 1]

0 means every client has the same label distribution; 1 means the client determines the
label. Computed over the 20 coarse superclasses, which is what the skew is drawn over.
`src/data.py:label_heterogeneity`.

**y-axis — drift.** At the start of a round, with every client evaluated at the *same*
broadcast model `w_t`, over its *whole* shard:

    g_i = grad F_i(w_t)                 exact, client i
    g   = sum_i (n_i/N) g_i             exact global gradient -- an identity, since the
                                        shards are a disjoint cover of the pool
    drift_t = (1/n) sum_i || g_i - g ||

Averaged over the rounds where it was measured. This is the bounded-gradient-dissimilarity
constant from the Local SGD convergence bounds. `src/server.py:measure_drift`.

Full-shard rather than mini-batch gradients on purpose: a sampled estimate would put a
noise floor under the measurement and the IID baseline would not come out near zero.

**Note on K.** Drift is measured at the broadcast point, so it does not depend on the
number of local steps. K=5 decides which points along the trajectory get measured, not
what is measured at them.

**Configuration.** small_cnn, full CIFAR-100, 20 clients, 100 rounds, K=5 local steps,
lr 0.05, batch 64, drift measured every 5 rounds (20 samples per run), loss every 10.
Heterogeneity swept as `iid`, `dirichlet` alpha in
{1000, 10, 3, 1, 0.5, 0.3, 0.1, 0.05, 0.01}, and `by_superclass`, each at seeds 0, 1, 2 --
33 runs. `by_superclass` ignores alpha and the rng, so its three seeds differ only in model
init and batch order, not in the partition.

**Covering the full [0, 1].** 20 clients over 20 superclasses is what makes H_label = 1
attainable, but Dirichlet alone does not get there: it never produces a clean
one-superclass-per-client permutation by chance, so it plateaus near 0.76 at alpha = 0.01
and starts emptying shards below that (alpha = 0.003 averages 0.80 with ~4 clients dropped).
The `by_superclass` partition deals superclasses out in contiguous blocks, giving exactly
one per client and H_label = 1 exactly. It anchors the top of the range; `iid` anchors the
bottom.

Measured over the full 50k training set, 20 clients, averaged over seeds 0-2:

    partition            H_label   empty shards
    iid                   0.0012        0
    dirichlet 1000        0.0002        0
    dirichlet 10          0.0147        0
    dirichlet 3           0.0462        0
    dirichlet 1           0.1286        0
    dirichlet 0.5         0.2172        0
    dirichlet 0.3         0.3081        0
    dirichlet 0.1         0.5047        0
    dirichlet 0.05        0.6299        0
    dirichlet 0.01        0.7574      1.3
    by_superclass         1.0000        0

Clients whose shard comes out empty are dropped with a warning, so alpha = 0.01 runs on
about 19 clients rather than 20. Drift is a mean over the surviving clients.

**Running.**

    jobs/drift_sweep.sh                    # sequential, local
    condor_submit jobs/drift_sweep.sub     # cluster, 33 parallel jobs

**Results.** One CSV per run in `runs/`, named
`small_cnn_<partition><alpha>_n20_K5_lr0.05_s<seed>.csv`, with columns
`round, grad_computations, train_loss, drift, grad_norm, h_label`. The drift and
grad_norm cells are empty on rows that were logged but not measured.

Gradient evaluations made for drift measurement are deliberately NOT counted in
`grad_computations`, which stays a pure optimization-cost axis.

**Plot.** `python plots/drift_vs_heterogeneity.py` -> `plots/drift_vs_heterogeneity.png`.
Two panels: absolute drift, and drift relative to ||g|| (which controls for gradients
shrinking as training converges).
