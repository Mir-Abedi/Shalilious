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
H_label swept over {0.00, 0.05, 0.10, ..., 1.00} at seeds 0, 1, 2 -- 63 runs.

**Sweeping H directly.** Rather than sweep Dirichlet alpha and measure whatever
heterogeneity comes out, `src/data.py:by_target_h` GENERATES a partition at a prescribed
H_label. Each client owns one of the 20 superclasses; a fraction t of every superclass's
samples goes to its owner and the rest is spread uniformly over all clients, so t=0 is IID
and t=1 is the pathological split. H_label is a closed form in t, strictly increasing, so t
is recovered from the requested H by bisection.

Two properties that make it the better axis:

- It hits the target to within 0.001 across [0, 1], so the x-axis is evenly spaced by
  construction instead of bunched wherever alpha happened to land.
- Every shard comes out EXACTLY the same size at every H. Under Dirichlet, skew and shard
  size grow together, so a drift-vs-heterogeneity trend is partly a drift-vs-shard-size
  trend. Here shard size is held fixed and heterogeneity is the only thing varying.

Requires n_clients to divide 20; the reachable ceiling is 1 - log(20/n_clients)/log(20), so
n_clients = 20 is what spans the full [0, 1]. A target above the ceiling raises rather than
silently missing.

`dirichlet` and `by_superclass` remain available as partitions; they are simply not what
this sweep varies.

**Running.**

    jobs/drift_sweep.sh                    # sequential, local
    condor_submit jobs/drift_sweep.sub     # cluster, 63 parallel jobs

**Results.** One CSV per run in `runs/`, named
`small_cnn_targeth<H>_n20_K5_lr0.05_s<seed>.csv`, with columns
`round, grad_computations, train_loss, drift, grad_norm, h_label`. The drift and
grad_norm cells are empty on rows that were logged but not measured.

Gradient evaluations made for drift measurement are deliberately NOT counted in
`grad_computations`, which stays a pure optimization-cost axis.

**Plot.** `python plots/drift_vs_heterogeneity.py` -> `plots/drift_vs_heterogeneity.png`.
Two panels: absolute drift, and drift relative to ||g|| (which controls for gradients
shrinking as training converges).
