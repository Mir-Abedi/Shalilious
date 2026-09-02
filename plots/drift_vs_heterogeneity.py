"""Drift vs. label heterogeneity.  Usage: python plots/drift_vs_heterogeneity.py

Reads every CSV in runs/ carrying drift measurements, groups the seeds at each
H_label, and plots the mean with +-1 sd error bars.

x = H_label, the partition's measured normalized mutual information I(C;Y)/H(Y).
y = client-gradient drift, averaged over the rounds where it was measured.
"""
import csv
import glob
import os
import statistics as st
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "plots", "drift_vs_heterogeneity.png")

# Validated palette: categorical slots 1 (blue) and 2 (orange), light mode.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def read_run(path):
    """(h_label, mean drift, mean drift/||g||, final train loss) for one run."""
    rows = list(csv.DictReader(open(path)))
    d = [(float(r["drift"]), float(r["grad_norm"])) for r in rows if r["drift"]]
    if not d:
        return None
    return (float(rows[0]["h_label"]),
            st.mean(x for x, _ in d),
            st.mean(x / y for x, y in d),
            float(rows[-1]["train_loss"]))


def main():
    runs = [r for r in (read_run(f) for f in glob.glob(os.path.join(ROOT, "runs", "*.csv"))) if r]
    if not runs:
        raise SystemExit("no runs/*.csv contained drift measurements -- run with --drift-every N")

    by_h = defaultdict(list)
    for h, a, rel, loss in runs:
        by_h[round(h, 3)].append((a, rel, loss))
    xs = sorted(by_h)
    n_seeds = min(len(by_h[x]) for x in xs)

    def stats(i):
        mean = [st.mean(v[i] for v in by_h[x]) for x in xs]
        sd = [st.stdev([v[i] for v in by_h[x]]) if len(by_h[x]) > 1 else 0.0 for x in xs]
        return mean, sd

    panels = [
        (r"mean $\|g_i - g\|$", "client-gradient drift", BLUE, 0),
        (r"mean $\|g_i - g\|\,/\,\|g\|$", "drift relative to the global gradient", BLUE, 1),
        ("final training loss", "what the drift costs", ORANGE, 2),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), facecolor=SURFACE)
    for ax, (ylab, title, color, idx) in zip(axes, panels):
        mean, sd = stats(idx)
        ax.errorbar(xs, mean, yerr=sd, color=color, linewidth=2, marker="o",
                    markersize=5, capsize=3, elinewidth=1.2, zorder=3)
        # endpoints only -- a number on every point is noise
        # endpoints sit above-left / below-right so neither label lands on the curve
        ax.annotate(f"{mean[0]:.2f}", (xs[0], mean[0]), textcoords="offset points",
                    xytext=(2, 12), fontsize=9, color=INK2)
        ax.annotate(f"{mean[-1]:.2f}", (xs[-1], mean[-1]), textcoords="offset points",
                    xytext=(9, -3), fontsize=9, color=INK2, ha="left")
        ax.set_xlim(-0.04, 1.16)   # room for the right-hand endpoint label
        ax.set_facecolor(SURFACE)
        ax.set_xlabel(r"$H_{\mathrm{label}} = I(C;Y)/H(Y)$", fontsize=10, color=INK)
        ax.set_ylabel(ylab, fontsize=10, color=INK)
        ax.set_title(title, fontsize=11, color=INK, pad=8)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.tick_params(colors=INK2, labelsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#d5d4cf")

    fig.suptitle("Client drift grows with label heterogeneity", fontsize=13, color=INK, y=0.99)
    fig.text(0.5, 0.005,
             f"CIFAR-100, small_cnn, 20 clients x 2500 samples each, K=5 local steps, "
             f"100 rounds, {n_seeds} seeds per point (error bars +-1 sd)",
             ha="center", fontsize=9, color=INK2)
    fig.tight_layout(rect=[0, 0.035, 1, 0.96])
    fig.savefig(OUT, dpi=160, facecolor=SURFACE)
    print(f"wrote {OUT}  ({len(runs)} runs, {len(xs)} H values, {n_seeds} seeds each)")


if __name__ == "__main__":
    main()
