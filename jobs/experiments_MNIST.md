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
-> `plots/drift_vs_heterogeneity_drift_small_cnn.png`. The figure is named after the run
dir; pass the dir explicitly, or it plots the CIFAR-100 runs.

### Results

_Pending — submitted 2026-09-02 on SaarlandHPC, condor cluster **185301**, 21 jobs x 3 seeds._
