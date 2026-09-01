"""Drift vs. label heterogeneity.  Usage: python plots/drift_vs_heterogeneity.py

Reads every CSV in runs/ that carries drift measurements and plots one point per run:
x = H_label (the partition's measured normalized mutual information I(C;Y)/H(Y)),
y = that run's client-gradient drift averaged over the rounds where it was measured.
"""
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "plots", "drift_vs_heterogeneity.png")


def read_run(path):
    """(h_label, mean drift, mean drift/||g||) for one run, or None if it has no drift rows."""
    h, drifts, rel = None, [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            if not row.get("drift"):
                continue
            h = float(row["h_label"])
            d, g = float(row["drift"]), float(row["grad_norm"])
            drifts.append(d)
            if g > 0:
                rel.append(d / g)
    if not drifts:
        return None
    return h, sum(drifts) / len(drifts), sum(rel) / len(rel) if rel else float("nan")


def main():
    points = sorted(
        p for p in (read_run(f) for f in glob.glob(os.path.join(ROOT, "runs", "*.csv"))) if p
    )
    if not points:
        raise SystemExit("no runs/*.csv contained drift measurements -- run with --drift-every N")

    xs = [p[0] for p in points]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, ys, label in (
        (axes[0], [p[1] for p in points], r"mean $\|g_i - g\|$"),
        (axes[1], [p[2] for p in points], r"mean $\|g_i - g\| / \|g\|$"),
    ):
        ax.scatter(xs, ys, s=34, alpha=0.8, edgecolor="none")
        ax.set_xlabel(r"$H_{\mathrm{label}} = I(C;Y)/H(Y)$")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3, linewidth=0.5)
    axes[0].set_title("client-gradient drift")
    axes[1].set_title("drift relative to the global gradient")
    fig.suptitle(f"Drift vs. label heterogeneity  ({len(points)} runs, K=5 local steps)")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}  ({len(points)} runs)")


if __name__ == "__main__":
    main()
