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

**Configuration.** small_cnn, full CIFAR-100, 10 clients, 100 rounds, K=5 local steps,
lr 0.05, batch 64, drift measured every 5 rounds (20 samples per run), loss every 10.
Heterogeneity swept as `iid` plus `dirichlet` alpha in
{1000, 10, 3, 1, 0.5, 0.3, 0.1, 0.05, 0.01}, each at seeds 0, 1, 2 -- 30 runs.

**Ceiling on the x-axis.** With 10 clients over 20 superclasses, even alpha -> 0 can only
give each client about 2 superclasses, so H_label saturates near 1 - ln(2)/ln(20) = 0.77;
alpha = 0.01 measures 0.68 in practice. The sweep therefore covers roughly [0, 0.7], not
the full [0, 1]. Reaching 1 would need at least 20 clients (one superclass each). Measured
values across the grid: alpha 1000 -> 0.000, 10 -> 0.017, 1 -> 0.123, 0.3 -> 0.292,
0.1 -> 0.456, 0.01 -> 0.677.

**Running.**

    jobs/drift_sweep.sh                    # sequential, local
    condor_submit jobs/drift_sweep.sub     # cluster, 30 parallel jobs

**Results.** One CSV per run in `runs/`, named
`small_cnn_<partition><alpha>_n10_K5_lr0.05_s<seed>.csv`, with columns
`round, grad_computations, train_loss, drift, grad_norm, h_label`. The drift and
grad_norm cells are empty on rows that were logged but not measured.

Gradient evaluations made for drift measurement are deliberately NOT counted in
`grad_computations`, which stays a pure optimization-cost axis.

**Plot.** `python plots/drift_vs_heterogeneity.py` -> `plots/drift_vs_heterogeneity.png`.
Two panels: absolute drift, and drift relative to ||g|| (which controls for gradients
shrinking as training converges).
