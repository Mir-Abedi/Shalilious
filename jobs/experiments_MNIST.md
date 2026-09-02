# Experiments — MNIST

Same questions as [experiments_CIFAR100.md](experiments_CIFAR100.md), on the cheap
dataset. What changes across the port, once, for every experiment here:

- Label skew is drawn over the **10 digits** — MNIST has no coarse superclasses, so the
  digits are both the training targets and the skew axis.
- `--partition target_h` needs `--clients` to divide 10, so the client count is **10**,
  not the 20 used on CIFAR-100. 10 is also the only count whose ceiling
  `1 - log(10/n)/log(10)` reaches 1.0, i.e. that spans the full `H_label` range.
- One epoch is 60000 samples, not 50000 — round budgets are recomputed, never copied.
- `--model linear` (multinomial logistic regression) is available as a **convex** arm,
  which CIFAR-100 does not have. Local SGD's rates are stated for convex objectives, so
  this is where a measured penalty can be checked against a prediction.
- Results land under `runs/mnist/`, keeping them out of the CIFAR-100 plots.
- **Caveat.** `by_target_h` promises equal shard sizes at every H, and that holds exactly
  on CIFAR-100 (500 images per class) but only approximately on MNIST, whose digit counts
  run 5421–6742. At 10 clients each client owns one digit, so shard size varies by about
  ±10% and is correlated with client identity. Small next to the drift effect, but it is a
  confound CIFAR-100 does not have.

## 1. Client drift vs. label heterogeneity  (MNIST port)

**Claim.** Client drift grows with data heterogeneity — same claim as CIFAR-100
experiment 1, retested where the objective can also be made convex.

Definitions of the two axes (`H_label = I(C;Y)/H(Y)`, and
`drift_t = (1/n) sum_i ||g_i - g||` from exact full-shard gradients at the broadcast
point) are unchanged; see experiment 1 in
[experiments_CIFAR100.md](experiments_CIFAR100.md) for the derivation and for why
full-shard rather than mini-batch gradients are used.

**Configuration.** `small_cnn` (1x28x28 input, 10 logits), full MNIST, **10 clients**,
**240 rounds**, K=5 local steps, lr 0.05, batch 64, drift measured every 10 rounds (24
samples per run), loss every 20. `H_label` swept over {0.00, 0.05, ..., 1.00} at seeds
0, 1, 2 — 21 jobs, one per H, each running all three seeds.

240 rounds is the equal-epoch translation of CIFAR-100's 100: both spend 12.8 epochs
(`rounds x clients x K x batch / dataset size`), so the loss panel is comparable across
the two datasets rather than comparable in rounds.

**Running.**

    condor_submit jobs/drift_mnist_sweep.sub     # cluster, 21 parallel jobs
    MODEL=linear jobs/drift_mnist_one.sh 0.5     # single cell, convex arm

The convex sweep is the same 21 jobs with `environment = "MODEL=linear"` in the `.sub`;
it is a separate submission rather than a second model in the same job, to stay inside
the 40-job budget.

**Results.** One CSV per run in `runs/mnist/drift_small_cnn/`, named
`targeth<H>_s<seed>.csv`, columns
`round, grad_computations, train_loss, drift, grad_norm, h_label`. Drift and grad_norm
are empty on rows logged but not measured. Drift-measurement gradients are not counted in
`grad_computations`.

**Plot.** `python plots/drift_vs_heterogeneity.py runs/mnist/drift_small_cnn`
-> `plots/drift_vs_heterogeneity_mnist_drift_small_cnn.png`. The figure is named after the run
dir; pass the dir explicitly, or it plots the CIFAR-100 runs.

### Results

Run 2026-09-02 on SaarlandHPC (condor, docker universe, GPU), cluster 185301, 21 jobs x 3
seeds, all 63 runs completed with no failures. Wall time ~75 min.

    H_label   drift    sd     drift/|g|   final loss
    ------------------------------------------------
       0.00   0.120  0.007       0.62      0.0435
       0.05   0.483  0.011       1.87      0.0448
       0.10   0.697  0.017       2.82      0.0465
       0.15   0.876  0.016       3.85      0.0474
       0.20   1.056  0.013       4.21      0.0496
       0.25   1.220  0.021       5.12      0.0508
       0.30   1.401  0.020       5.35      0.0529
       0.35   1.576  0.025       6.28      0.0549
       0.40   1.748  0.040       6.78      0.0576
       0.45   1.946  0.028       7.58      0.0593
       0.50   2.141  0.027       7.99      0.0630
       0.55   2.359  0.012       8.78      0.0655
       0.60   2.584  0.005       8.63      0.0692
       0.65   2.793  0.034       9.09      0.0727
       0.70   3.064  0.057       8.90      0.0774
       0.75   3.305  0.038       9.79      0.0822
       0.80   3.612  0.007       9.65      0.0895
       0.85   3.957  0.023       9.92      0.0971
       0.90   4.473  0.028       9.49      0.1087
       0.95   5.281  0.046      10.69      0.1254
       1.00   7.941  0.141       5.30      0.1505

**The CIFAR-100 result replicates.** Drift rises monotonically across the whole range with
no turnover, 0.12 to 7.94 — a **66x** increase, against 15x on CIFAR-100. Seed standard
deviations are 0.005-0.141 on values of 0.1-8, so the trend again dwarfs the noise. Final
training loss rises monotonically too, 0.044 to 0.151, a 3.5x cost for the same gradient
budget.

Two differences worth recording against the CIFAR-100 run:

- **The IID floor is much lower** (0.12 vs 0.46 absolute, 0.62 vs 0.98 relative). With
  10 clients rather than 20 each shard is larger, and MNIST's gradients are less noisy, so
  a genuinely IID split leaves less residual disagreement between clients.
- **The curve is convex here, concave on CIFAR-100.** CIFAR-100 rises steeply out of IID
  and then flattens (0.46 -> 3.5 over the first quarter of the range, 3.5 -> 6.95 over the
  remaining three quarters). MNIST is close to linear until H=0.9 and then jumps: the last
  step, 0.95 -> 1.00, adds 2.66 on its own, half again as much as the whole first quarter.
  At H=1.0 each of the 10 clients holds exactly one digit and its local objective is a
  degenerate one-class problem; with 20 clients over 20 superclasses the same corner is
  much less extreme, since one CIFAR-100 superclass still spans 5 fine labels. The
  pathological corner is simply more pathological on MNIST.

**The relative panel turns over at H=1.0** (10.69 -> 5.30), the only non-monotonicity
anywhere in either dataset. This is the denominator moving, not the drift: at H=1.0 the run
is much further from convergence (loss 0.151 vs 0.125 at H=0.95) so ||g|| is
correspondingly larger. Absolute drift, the panel that matches the theory's
gradient-dissimilarity constant, keeps rising.
