"""Plot templates for the four focused notebook studies.

The palette, light surface, typography, mean curves, and +-1 sd uncertainty follow the
project's existing ``drift_vs_heterogeneity.py`` and ``loss_vs_sync_interval.py``.
"""
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


COLORS = ["#2a78d6", "#eb6834", "#7b53c1", "#1baf7a", "#c03d73"]
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.tick_params(colors=INK2, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d5d4cf")


def _mean_sd(rows, x_key, y_key):
    values = defaultdict(list)
    for row in rows:
        if row.get(y_key) is not None:
            values[row[x_key]].append(float(row[y_key]))
    xs = sorted(values)
    means = np.asarray([np.mean(values[x]) for x in xs])
    sds = np.asarray([np.std(values[x], ddof=1) if len(values[x]) > 1 else 0 for x in xs])
    return np.asarray(xs), means, sds


def hard_client_plot(histories):
    """Round-by-round global loss for each noisy-client local-step setting."""
    hard_steps = sorted({row["hard_k"] for row in histories})
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.2), facecolor=SURFACE)
    for color, hard_k in zip(COLORS, hard_steps):
        rows = [row for row in histories if row["hard_k"] == hard_k]
        rounds, mean, sd = _mean_sd(rows, "round", "train_loss")
        ax.plot(rounds, mean, color=color, linewidth=2,
                label=rf"$\tau_{{\mathrm{{noisy}}}}={hard_k}$")
        ax.fill_between(rounds, mean - sd, mean + sd,
                        color=color, alpha=0.14, linewidth=0)
    ax.set_xlabel("Communication Round", color=INK)
    ax.set_ylabel("Global Mixed Training Loss", color=INK)
    ax.legend(frameon=False, ncol=2)
    _style(ax)
    fig.tight_layout()
    return fig


def budget_curve_plot(histories, *, group_key, group_order, group_labels,
                      facet_key=None, facet_order=(None,), facet_labels=None, title=""):
    """Mean training loss against rounds and actual local-update computation."""
    facet_labels = facet_labels or {None: ""}
    fig, axes = plt.subplots(len(facet_order), 2, squeeze=False,
                             figsize=(10, 3.7 * len(facet_order)), facecolor=SURFACE,
                             sharex="col")
    for row_index, facet in enumerate(facet_order):
        facet_rows = [r for r in histories if facet_key is None or r[facet_key] == facet]
        for color, group in zip(COLORS, group_order):
            rows = [r for r in facet_rows if r[group_key] == group]
            for column, x_key in enumerate(("round", "local_updates")):
                x, mean, sd = _mean_sd(rows, x_key, "train_loss")
                axes[row_index, column].plot(x, mean, color=color, linewidth=2,
                                              label=group_labels[group])
                axes[row_index, column].fill_between(x, mean - sd, mean + sd,
                                                      color=color, alpha=0.14, linewidth=0)
        axes[row_index, 0].set_ylabel("Training Loss", color=INK)
        axes[row_index, 0].set_title(
            f"{facet_labels[facet]} — communication budget" if facet_labels[facet]
            else "Communication budget", color=INK, fontsize=11
        )
        axes[row_index, 1].set_title(
            f"{facet_labels[facet]} — computation budget" if facet_labels[facet]
            else "Computation budget", color=INK, fontsize=11
        )
        for ax in axes[row_index]:
            _style(ax)
    axes[-1, 0].set_xlabel("communication rounds", color=INK)
    axes[-1, 1].set_xlabel("client mini-batch gradient updates", color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4),
               frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(title, color=INK, fontsize=13)
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    return fig
