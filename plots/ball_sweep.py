"""Trust-region Local SGD.  Usage: python plots/ball_sweep.py

Writes two figures:
  ball_loss.png         training loss, rows = heterogeneity, cols = K, one curve per rho
  ball_diagnostics.png  how often the ball bound, and how well clients still descended

rho is an ordered parameter, so it gets a sequential ramp (light = tight ball, dark =
loose), with the rho=inf control in grey -- it is a different thing, not a further step.
"""
import csv
import glob
import os
import sys
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")

# The two sweeps differ in grid: CIFAR-100 ran K<=50 and rho from 0.25, MNIST runs K to 150
# and rho from 0.5. Cells with no CSVs are simply skipped, so one union grid serves both.
KS = [1, 5, 10, 20, 40, 50, 70, 100, 150]
RHOS = ["0.25", "0.5", "1", "2", "4", "8", "inf"]
# sequential blue ramp for the ordered rho, grey for the no-ball control
RAMP = {"0.25": "#cfe3f8", "0.5": "#a9cdf1", "1": "#7db0e8", "2": "#4a8ddc", "4": "#2364b8",
        "8": "#143f77", "inf": "#8a8983"}
HS = ["1.0", "0.75"]
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def load(k, rho, h):
    """Mean across seeds of every logged column, keyed by gradient computations."""
    acc = defaultdict(lambda: defaultdict(list))
    for p in glob.glob(os.path.join(BALL, f"K{k}_rho{rho}_h{h}_s*.csv")):
        for r in csv.DictReader(open(p)):
            g = int(r["grad_computations"])
            for col in ("train_loss", "ball_hits", "cos_client", "cos_agg", "mean_hit_step"):
                if r.get(col):
                    acc[col][g].append(float(r[col]))
    return {c: (sorted(d), [sum(d[x]) / len(d[x]) for x in sorted(d)]) for c, d in acc.items()}


def style(ax, xlab, ylab, title):
    ax.set_facecolor(SURFACE)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.tick_params(colors=INK2, labelsize=9)
    if xlab:
        ax.set_xlabel(xlab, fontsize=9, color=INK)
    if ylab:
        ax.set_ylabel(ylab, fontsize=9, color=INK)
    if title:
        ax.set_title(title, fontsize=10, color=INK, pad=6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d5d4cf")


def figure_loss(data, ks, rhos):
    fig, axes = plt.subplots(2, len(ks), figsize=(3.4 * len(ks), 7), sharex=True,
                             sharey=True, squeeze=False, facecolor=SURFACE)
    for row, h in enumerate(HS):
        for col, k in enumerate(ks):
            ax = axes[row, col]
            for rho in rhos:
                d = data.get((k, rho, h), {})
                if "train_loss" in d:
                    xs, ys = d["train_loss"]
                    ax.plot(xs, ys, color=RAMP[rho], linewidth=2,
                            linestyle="--" if rho == "inf" else "-",
                            label=(r"$\rho=\infty$ (no ball)" if rho == "inf" else rf"$\rho={rho}$"),
                            zorder=3 if rho != "inf" else 2)
            style(ax, "gradient computations" if row == 1 else None,
                  f"H = {h}\ntraining loss" if col == 0 else None,
                  f"K = {k}" if row == 0 else None)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Confining local steps to a ball of radius "
                 r"$R_t=\rho\,\|g_t\|\,\eta K$", fontsize=13, color=INK, y=0.985)
    fig.tight_layout(rect=[0, 0.06, 1, 0.955])
    out = os.path.join(ROOT, "plots", f"ball_loss{TAG}.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


def figure_diagnostics(data, ks, rhos):
    # The client count is whatever this sweep ran; read it off a ball_hits series rather
    # than hardcoding the 20 the CIFAR-100 sweep used.
    n_clients = max((max(v["ball_hits"][1]) for v in data.values()
                     if "ball_hits" in v and v["ball_hits"][1]), default=0)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), squeeze=False, facecolor=SURFACE)
    for row, h in enumerate(HS):
        for col, (metric, ylab) in enumerate(
                [("ball_hits", f"clients hitting the ball (of {n_clients})"),
                 ("cos_client", r"$\cos(\Delta_i,\,-g_t)$")]):
            ax = axes[row, col]
            for rho in rhos:
                xs, ys = [], []
                for k in ks:
                    d = data.get((k, rho, h), {})
                    if metric in d and d[metric][1]:
                        xs.append(k)
                        ys.append(sum(d[metric][1]) / len(d[metric][1]))
                if xs:
                    ax.plot(xs, ys, color=RAMP[rho], linewidth=2, marker="o", markersize=5,
                            linestyle="--" if rho == "inf" else "-",
                            label=(r"$\rho=\infty$" if rho == "inf" else rf"$\rho={rho}$"))
            ax.set_xscale("log")
            ax.set_xticks(KS)
            ax.set_xticklabels(KS)
            style(ax, "local steps K" if row == 1 else None, ylab,
                  f"H = {h}" if col == 0 else None)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("How often the ball binds, and whether clients still descend",
                 fontsize=13, color=INK, y=0.985)
    fig.tight_layout(rect=[0, 0.06, 1, 0.955])
    out = os.path.join(ROOT, "plots", f"ball_diagnostics{TAG}.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


def main():
    global BALL, TAG
    BALL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "runs", "ball")
    if not os.path.isdir(BALL):
        raise SystemExit(f"no {BALL}/ -- run the sweep first")
    # Figures are named after the run dir so a second dataset does not overwrite the first.
    rel = os.path.relpath(os.path.normpath(BALL), os.path.join(ROOT, "runs"))
    TAG = "" if rel == "ball" else "_" + rel.replace(os.sep, "_")
    data = {(k, r, h): load(k, r, h) for k in KS for r in RHOS for h in HS}
    data = {key: v for key, v in data.items() if v}
    if not data:
        raise SystemExit(f"no CSVs in {BALL}/")
    # Only draw the K and rho values this sweep actually produced.
    present_k = sorted({k for k, _, _ in data}, key=KS.index)
    present_r = sorted({r for _, r, _ in data}, key=RHOS.index)
    figure_loss(data, present_k, present_r)
    figure_diagnostics(data, present_k, present_r)


if __name__ == "__main__":
    main()
