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


def hard_client_plot(summaries):
    """Domain loss and source-defined drift as the hard client's K changes."""
    expanded = []
    for row in summaries:
        item = dict(row)
        item["easy_loss"] = row["final_client_losses"][0]
        item["hard_loss"] = row["final_client_losses"][1]
        expanded.append(item)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), facecolor=SURFACE)
    for key, label, color in [
        ("easy_loss", "clean client", COLORS[0]),
        ("hard_loss", "corrupted client", COLORS[1]),
    ]:
        x, mean, sd = _mean_sd(expanded, "hard_k", key)
        axes[0].errorbar(x, mean, yerr=sd, marker="o", capsize=3, linewidth=2,
                         color=color, label=label)
    x, mean, sd = _mean_sd(expanded, "hard_k", "final_drift")
    axes[1].errorbar(x, mean, yerr=sd, marker="o", capsize=3, linewidth=2,
                     color=COLORS[2])
    axes[0].set(xlabel=r"hard-client local steps $K_{hard}$",
                ylabel="final client training loss", title="Which domain benefits?")
    axes[1].set(xlabel=r"hard-client local steps $K_{hard}$",
                ylabel=r"mean $\|g_i-g\|$", title="Gradient drift at the broadcast point")
    axes[0].legend(frameon=False)
    for ax in axes:
        _style(ax)
    fig.suptitle("Extra local work on a covariate-shifted client", color=INK, fontsize=13)
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
        axes[row_index, 0].set_ylabel("training loss", color=INK)
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
