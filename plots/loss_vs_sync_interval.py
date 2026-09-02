"""Train loss vs. synchronization interval.  Usage: python plots/loss_vs_sync_interval.py

One panel per K (local steps between syncs), three curves per panel (heterogeneity
levels), and centralized SGD as the dashed ceiling on every panel. All panels share
axes -- that is what makes the widening gap readable across the grid.
"""
import csv
import glob
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
SYNC = os.path.join(ROOT, "runs", "sync")
OUT = os.path.join(ROOT, "plots", "loss_vs_sync_interval.png")

KS = [1, 5, 10, 20, 40, 70, 100, 150]
# Validated palette, categorical slots 1-3 (all-pairs safe at three series).
HCOLOR = {"0.0": "#2a78d6", "0.5": "#eb6834", "1.0": "#1baf7a"}
HLABEL = {"0.0": "H = 0.0  (homogeneous)", "0.5": "H = 0.5  (semi)", "1.0": "H = 1.0  (heterogeneous)"}
INK, INK2, SURFACE, REF = "#0b0b0b", "#52514e", "#fcfcfb", "#8a8983"


def curve(paths):
    """Mean loss across seeds at each logged point: ([grads], [loss])."""
    acc = defaultdict(list)
    for p in paths:
        for r in csv.DictReader(open(p)):
            acc[int(r["grad_computations"])].append(float(r["train_loss"]))
    xs = sorted(acc)
    return xs, [sum(acc[x]) / len(acc[x]) for x in xs]


def main():
    if not os.path.isdir(SYNC):
        raise SystemExit(f"no {SYNC}/ -- run jobs/sync_sweep.sub first")
    ref = curve(glob.glob(os.path.join(SYNC, "central_s*.csv")))

    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharex=True, sharey=True,
                             facecolor=SURFACE)
    seen = 0
    for ax, k in zip(axes.ravel(), KS):
        for h in ("0.0", "0.5", "1.0"):
            paths = glob.glob(os.path.join(SYNC, f"K{k}_h{h}_s*.csv"))
            if not paths:
                continue
            xs, ys = curve(paths)
            ax.plot(xs, ys, color=HCOLOR[h], linewidth=2, label=HLABEL[h], zorder=3)
            seen += 1
        if ref[0]:
            ax.plot(ref[0], ref[1], color=REF, linewidth=1.6, linestyle="--",
                    label="centralized SGD", zorder=2)
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
        raise SystemExit(f"no K*_h*_s*.csv in {SYNC}/")

    for ax in axes[1]:
        ax.set_xlabel("gradient computations", fontsize=10, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("training loss", fontsize=10, color=INK)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Longer synchronization intervals stall convergence, and heterogeneity "
                 "makes it worse", fontsize=13, color=INK, y=0.985)
    fig.text(0.5, 0.055, "CIFAR-100, small_cnn, 20 clients x 2500 samples each, equal "
             "gradient budget of 100 epochs per panel, 3 seeds, lr 0.05, batch 64",
             ha="center", fontsize=9, color=INK2)
    fig.tight_layout(rect=[0, 0.085, 1, 0.955])
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
