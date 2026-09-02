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
H_label swept over {0.00, 0.05, 0.10, ..., 1.00} at seeds 0, 1, 2 -- 21 jobs, one per H, each running all three seeds.

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
    condor_submit jobs/drift_sweep.sub     # cluster, 21 parallel jobs

**Results.** One CSV per run in `runs/`, named
`small_cnn_targeth<H>_n20_K5_lr0.05_s<seed>.csv`, with columns
`round, grad_computations, train_loss, drift, grad_norm, h_label`. The drift and
grad_norm cells are empty on rows that were logged but not measured.

Gradient evaluations made for drift measurement are deliberately NOT counted in
`grad_computations`, which stays a pure optimization-cost axis.

**Plot.** `python plots/drift_vs_heterogeneity.py` -> `plots/drift_vs_heterogeneity.png`.
Three panels: absolute drift, drift relative to ||g|| (which controls for gradients
shrinking as training converges), and the final training loss that the drift costs.

### Results

Run 2026-09-02 on SaarlandHPC (condor, docker universe, Tesla P100), 21 jobs x 3 seeds,
all 63 runs completed with no failures.

    H_label   drift    sd     drift/|g|   final loss
    ------------------------------------------------
       0.00   0.464  0.008       0.98      3.4311
       0.05   1.450  0.030       2.96      3.4223
       0.10   2.155  0.021       4.31      3.4378
       0.15   2.694  0.020       5.25      3.4566
       0.20   3.149  0.018       6.08      3.4818
       0.25   3.539  0.010       6.72      3.5088
       0.30   3.868  0.021       7.35      3.5362
       0.35   4.161  0.021       7.89      3.5622
       0.40   4.463  0.019       8.39      3.5873
       0.45   4.718  0.051       8.82      3.6055
       0.50   4.947  0.015       9.14      3.6397
       0.55   5.168  0.038       9.51      3.6663
       0.60   5.398  0.033       9.93      3.7001
       0.65   5.608  0.073      10.25      3.7256
       0.70   5.843  0.039      10.65      3.7583
       0.75   5.935  0.057      10.83      3.7966
       0.80   6.178  0.040      11.20      3.8294
       0.85   6.321  0.068      11.58      3.8737
       0.90   6.469  0.075      11.91      3.8989
       0.95   6.726  0.074      12.39      3.9394
       1.00   6.953  0.041      12.72      3.9709

**Drift rises monotonically with heterogeneity across the whole range** -- 0.46 to 6.95, a
15x increase, with no turnover anywhere. The relative measure rises 0.98 to 12.72. Seed
standard deviations are 0.008-0.075 on values of 0.5-7, i.e. under 1.5%, so the trend is
two orders of magnitude larger than the noise.

Three things the sweep settles:

- **The shape is concave, not linear.** Drift jumps 0.46 -> 2.16 over the first 0.1 of
  H_label, then climbs at a decreasing rate. A little heterogeneity costs a lot; further
  heterogeneity costs progressively less.
- **Optimization degrades in lockstep.** Final training loss rises monotonically from
  3.43 to 3.97 across the same range, at identical gradient budget, client count and
  shard size. This is the consequence of the drift, and it completes the argument: skew
  raises gradient dissimilarity, and dissimilarity slows optimization.
- **Shard size is ruled out as the cause.** Every client holds exactly 2500 samples at
  every H (verified in the job logs: spread 0 across all 63 runs), so nothing but label
  composition varies along the x-axis.

A pilot on synthetic data had suggested absolute drift might turn over near H = 1 while
the relative measure kept rising. That did not reproduce on CIFAR-100 -- both measures are
monotone to H = 1. The `grad_norm` column stays worth logging, but the concern it was
guarding against did not materialise.

## 2. Train loss vs. synchronization interval

**Claim.** Synchronizing less often stalls convergence, and heterogeneity multiplies the
damage.

**Design.** Every cell spends the same gradient budget -- 100 epochs = 5e6 sample-gradients
-- so rounds shrink as K grows and the only variable is how often clients synchronize. Any
loss difference across panels is the cost of syncing less, not a difference in compute.

    K        1     5    10    20    40    70   100   150
    rounds 3906   781   391   195    98    56    39    26

**Centralized SGD reference.** Local SGD with ONE client is exactly centralized SGD:
aggregation over a single client is the identity and no drift is possible. So the
reference needs no new code -- it is `--clients 1 --local-steps 1000 --rounds 78
--batch-size 64`, plain batch-64 SGD over the same 5e6 budget. It has no clients and
therefore no heterogeneity, so one run serves as the ceiling on all 8 panels.

**Configuration.** small_cnn, full CIFAR-100, 20 clients x 2500 samples each, lr 0.05,
batch 64, H_label in {0.0, 0.5, 1.0} via `by_target_h`, 3 seeds -- 24 jobs plus 1 for the
reference.

**Running.**

    condor_submit jobs/sync_sweep.sub      # cluster, 24 jobs
    condor_submit jobs/sgd_baseline.sub    # the centralized reference

**Results.** CSVs in `runs/sync/`, named `K<K>_h<H>_s<seed>.csv` and `central_s<seed>.csv`
(kept out of `runs/` proper so they do not collide with experiment 1's K=5 filenames or
enter its plot glob). Run 2026-09-02, all 75 runs completed with no failures.

Final training loss after 100 epochs (mean of 3 seeds +- sd):

        K      H=0.0             H=0.5             H=1.0
    --------------------------------------------------------
        1   0.8536 +-0.0824    0.7588 +-0.0703    0.7108 +-0.0213  
        5   1.1120 +-0.0314    1.8326 +-0.0563    2.5437 +-0.0450  
       10   1.2110 +-0.0382    1.9140 +-0.0532    2.7803 +-0.0459  
       20   1.2956 +-0.0381    1.9513 +-0.0573    2.9835 +-0.0405  
       40   1.4710 +-0.0499    1.9931 +-0.0581    3.1677 +-0.0395  
       70   1.4620 +-0.0732    1.9512 +-0.0493    3.2519 +-0.0466  
      100   1.4068 +-0.0497    1.8897 +-0.0564    3.3029 +-0.0285  
      150   1.4978 +-0.0606    1.9286 +-0.0283    3.3730 +-0.0365  

    centralized SGD (batch 64): 0.0005 +-0.0000

- **At K=1 heterogeneity costs nothing** -- 0.85 / 0.76 / 0.71 are within noise of each
  other, and if anything H=1.0 is lowest. This is not luck: one local step followed by a
  shard-size-weighted average IS exact large-batch SGD, so the partition cannot matter.
  It is the same identity `test_one_local_step_matches_plain_sgd` asserts, and it makes
  the effect at larger K attributable to local drift rather than to the setup.
- **The interaction is the result, not the main effect.** Going K=1 -> 150 costs 0.64 loss
  on homogeneous data but 2.66 on fully heterogeneous data -- four times the penalty.
  Syncing rarely is nearly free when clients agree and ruinous when they do not.
- **Under heterogeneity the loss never stops degrading with K** (2.54 -> 3.37 from K=5 to
  150, still climbing). On homogeneous data it plateaus around 1.4-1.5 past K=40. So
  there is a usable ceiling on local work when data is IID, and none when it is skewed.
- **The distributed-vs-centralized gap dwarfs all of it.** Plain batch-64 SGD reaches
  5e-4 while the best distributed arm reaches 0.71 -- three orders of magnitude, at an
  identical gradient budget. The cause is parameter updates, not gradients: centralized
  SGD takes 78,125 steps of batch 64, while K=1 Local SGD takes 3,906 steps of effective
  batch 1280. Averaging 20 clients buys a less noisy gradient and pays 20x fewer updates,
  and at this budget that trade is heavily negative.

**Plot.** `python plots/loss_vs_sync_interval.py` -> `plots/loss_vs_sync_interval.png`.
One panel per K, one curve per heterogeneity level, centralized SGD dashed on every panel.
Log y-axis: the reference reaches 5e-4 and a linear axis flattens it onto the baseline.


## 3. Trust-region Local SGD: bounding how far clients travel

**Idea.** K bounds how *long* clients run between syncs; it does not bound how *far* they
go. Bound the distance directly. Each round the server sets

    R_t = rho * ||g_t|| * eta * K

and confines every client's local iterates to the ball B(w_t, R_t). A tentative step
u = w - eta*g that leaves the ball is projected onto its surface,

    w <- w_t + min(1, R_t / ||u - w_t||) * (u - w_t)

after which that client stops for the round. rho is the free parameter: rho -> 0 is
one-shot averaging, rho = infinity is ordinary Local SGD.

**Estimating ||g_t||.** The exact global gradient costs a full pass over the training set
every round -- roughly 40x the whole training budget at K=1 -- so the radius is scaled by
an unbiased 2048-sample minibatch gradient taken at w_t before the clients are dispatched.
Overhead is 1/K of a round's training work (100% at K=1, 2% at K=50). The radius only
needs the scale of ||g_t||, and rho sweeps a 16x range around it.

**Logged per round.** `ball_hits` (how many of the 20 clients hit the ball),
`mean_hit_step` (the local step at which they hit), `cos_client` (mean over clients of
cos(w_i,K - w_t, -g_t)), `cos_agg` (the same for the aggregated update), `radius`, and
`train_loss`. The cosines use the same 2048-sample estimate; `cos_client_exact` recomputes
against the true full-batch gradient on every 10th logged round as a check.

**Configuration.** small_cnn, full CIFAR-100, 20 clients x 2500 samples each, lr 0.05,
batch 64, equal gradient budget of 100 epochs -- the same budget as experiment 2, so the
rho=infinity column is directly comparable to those results and acts as the control.
Swept over rho in {0.25, 0.5, 1, 2, 4, inf}, K in {1, 5, 10, 20, 50}, and H_label in
{1.0 (fully heterogeneous), 0.75 (almost)}, 3 seeds each -- 180 runs packed into 30 jobs,
one per (rho, K).

rho=infinity is implemented as radius=None, which delegates to the ordinary
`Client.local_steps`, so the control arm is the same code path rather than a second
implementation that could drift from it.

**Running.**

    condor_submit jobs/ball_sweep.sub     # cluster, 30 parallel jobs
    jobs/ball_one.sh <rho> <K>            # one cell, locally

**Results.** One CSV per run in `runs/ball/`, named `K<K>_rho<rho>_h<H>_s<seed>.csv`.

**Plots.** `python plots/ball_sweep.py` -> `plots/ball_loss.png` (loss, rows =
heterogeneity, cols = K, one curve per rho) and `plots/ball_diagnostics.png` (how often
the ball bound, and whether clients still descended).

**Early note from the smoke run.** On a reduced-data smoke, every client hit the ball by
local step ~2 of 10 at rho=1 and at step 1 at rho=0.25. Under heterogeneity a client's own
mini-batch gradient is much larger than the global ||g_t|| that sizes the ball, so the
tight rho values may collapse onto K=1 behaviour and the contrast may sit between rho=4
and rho=infinity.
