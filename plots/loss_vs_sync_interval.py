"""Train loss vs. synchronization interval.

    python plots/loss_vs_sync_interval.py [run_dir] [x_axis]

One panel per K (local steps between syncs), three curves per panel (heterogeneity
levels), and centralized SGD as the ceiling on every panel. All panels share axes --
that is what makes the widening gap readable across the grid.

Two sweep designs are supported, distinguished by what the reference CSVs are called:

  equal gradient budget (CIFAR-100)   rounds shrink as K grows, every panel spends the
      same compute, so ONE pair of centralized references serves the whole grid
      (central_s*.csv at batch 64, matched_s*.csv at the effective batch). Plot against
      gradient computations: that is the shared axis.

  equal communication budget (MNIST)  rounds are pinned, so compute grows with K and no
      single reference applies. Each panel carries its own ceiling at matched compute
      (centralK<k>_s*.csv). Plot against rounds: that is the shared axis.

x_axis is "grads" (default) or "rounds".
"""
import csv
import glob
import os
import re
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")

KS = [1, 5, 10, 20, 40, 70, 100, 150]
# Validated palette, categorical slots 1-3 (all-pairs safe at three series).
HCOLOR = {"0.0": "#2a78d6", "0.5": "#eb6834", "1.0": "#1baf7a"}
HLABEL = {"0.0": "H = 0.0  (homogeneous)", "0.5": "H = 0.5  (semi)", "1.0": "H = 1.0  (heterogeneous)"}
INK, INK2, SURFACE, REF = "#0b0b0b", "#52514e", "#fcfcfb", "#8a8983"


def curve(paths, xcol="grad_computations"):
    """Mean loss across seeds at each logged point: ([x], [loss]).

    Runs that stopped early are dropped rather than averaged in: a seed still in
    progress carries a mid-run loss that would silently drag the tail of the mean.
    """
    runs = []
    for p in paths:
        rows = [(int(r[xcol]), float(r["train_loss"]))
                for r in csv.DictReader(open(p))]
        if rows:
            runs.append(rows)
    if not runs:
        return [], []
    reach = max(r[-1][0] for r in runs)
    runs = [r for r in runs if r[-1][0] >= 0.9 * reach]
    acc = defaultdict(list)
    for r in runs:
        for g, l in r:
            acc[g].append(l)
    xs = sorted(acc)
    return xs, [sum(acc[x]) / len(acc[x]) for x in xs]


def main():
    sync = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "runs", "sync")
    xcol = "round" if (len(sys.argv) > 2 and sys.argv[2] == "rounds") else "grad_computations"
    if not os.path.isdir(sync):
        raise SystemExit(f"no {sync}/ -- run the sweep first")
    tag = os.path.relpath(os.path.normpath(sync), os.path.join(ROOT, "runs"))
    out = os.path.join(ROOT, "plots", "loss_vs_sync_interval"
                       + ("" if tag == "sync" else "_" + tag.replace(os.sep, "_")) + ".png")

    ref = curve(glob.glob(os.path.join(sync, "central_s*.csv")), xcol)
    matched = curve(glob.glob(os.path.join(sync, "matched_s*.csv")), xcol)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharex=True, sharey=True,
                             facecolor=SURFACE)
    seen = 0
    for ax, k in zip(axes.ravel(), KS):
        for h in ("0.0", "0.5", "1.0"):
            paths = glob.glob(os.path.join(sync, f"K{k}_h{h}_s*.csv"))
            if not paths:
                continue
            xs, ys = curve(paths, xcol)
            ax.plot(xs, ys, color=HCOLOR[h], linewidth=2, label=HLABEL[h], zorder=3)
            seen += 1
        # Per-panel ceiling, when the sweep provides one: centralized SGD at this panel's
        # own compute. Falls back to the single grid-wide pair for the equal-budget design.
        per_panel = curve(glob.glob(os.path.join(sync, f"centralK{k}_s*.csv")), xcol)
        if per_panel[0]:
            ax.plot(per_panel[0], per_panel[1], color="#4a3aa7", linewidth=1.8,
                    linestyle=":", label="centralized SGD at matched compute", zorder=4)
        elif matched[0]:
            # the like-for-like control: batch 1280 is exactly K=1's effective batch
            ax.plot(matched[0], matched[1], color="#4a3aa7", linewidth=1.8, linestyle=":",
                    label="centralized SGD, batch 1280", zorder=4)
        if ref[0]:
            ax.plot(ref[0], ref[1], color=REF, linewidth=1.6, linestyle="--",
                    label="centralized SGD, batch 64", zorder=2)
        ax.set_yscale("log")   # centralized SGD reaches 5e-4; linear would flatten it
        ax.set_title(f"K = {k}", fontsize=11, color=INK, pad=6)
        ax.set_facecolor(SURFACE)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.tick_params(colors=INK2, labelsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#d5d4cf")
    if not seen:
        raise SystemExit(f"no K*_h*_s*.csv in {sync}/")

    xname = "communication rounds" if xcol == "round" else "gradient computations"
    for ax in axes[1]:
        ax.set_xlabel(xname, fontsize=10, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("training loss", fontsize=10, color=INK)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 0.045))
    fig.suptitle("Longer synchronization intervals stall convergence, and heterogeneity "
                 "makes it worse", fontsize=13, color=INK, y=0.985)
    caption = ("CIFAR-100, small_cnn, 20 clients x 2500 samples each, equal gradient "
               "budget of 100 epochs per panel, 3 seeds, lr 0.05, client batch 64. K=1 "
               "with 20 clients has effective batch 1280, so the batch-1280 line is its "
               "like-for-like control." if tag == "sync" else
               f"{tag}, small_cnn, 10 clients, 100 rounds in every panel, 3 seeds, lr 0.05, "
               "client batch 64. Compute grows with K, so each panel carries its own "
               "centralized ceiling at that panel's budget (batch 640, the effective batch "
               "of 10 clients x 64).")
    fig.text(0.5, 0.008, caption, ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=[0, 0.10, 1, 0.955])
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
