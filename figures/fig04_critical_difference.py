"""Figure 4: critical difference diagrams from a Nemenyi post-hoc test.

Each stream is one problem and the methods are ranked within it, so rank 1 is the
best. A Friedman test is run first; the horizontal bars then join the methods
whose average ranks are not significantly different at the given level. Four
panels: accuracy and macro-F1, each at 5% and 1% labels.

Only streams where every method completed are used, so all methods are ranked
over the same set.

Reads: results/ablation/runs.csv
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare, rankdata

from _benchmark import (CONFIGS, PLAIN_NAMES, STREAMS, cell_matrix, load)
from _common import BENCHMARK_CSV, INK, save

ALPHA = 0.10
TARGETS = [("acc", "Accuracy"), ("f1", "Macro-F1")]
LABEL_PCTS = [1, 5]


def _table(cells, label_pct, metric):
    """Streams by methods, restricted to streams where every method ran."""
    matrix = cell_matrix(cells, metric, label_pct)
    frame = pd.DataFrame(matrix.T, columns=[PLAIN_NAMES[c] for c in CONFIGS],
                         index=[s for _, s in STREAMS])
    return frame.dropna()


def _panel(ax, table, title):
    ranks = table.apply(lambda row: rankdata(-row.values), axis=1,
                        result_type="expand")
    ranks.columns = table.columns
    average = ranks.mean(axis=0)
    _, p = friedmanchisquare(*[table[c].values for c in table.columns])
    significance = sp.posthoc_nemenyi_friedman(table.values)
    significance.index = table.columns
    significance.columns = table.columns

    plt.sca(ax)
    sp.critical_difference_diagram(
        average, significance, ax=ax, alpha=ALPHA,
        label_fmt_left="{label}  ", label_fmt_right="  {label}",
        label_props={"fontsize": 138, "fontweight": "bold"},
        crossbar_props={"linewidth": 16.0},
        marker_props={"s": 7000},
        elbow_props={"linewidth": 12.0})
    ax.set_title(title, fontsize=158, fontweight="bold", color=INK, pad=92)
    ax.tick_params(axis="x", labelsize=130, width=10.0, length=38)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    return average, p


def main():
    cells = load()
    combos = [(metric, name, lp) for metric, name in TARGETS
              for lp in LABEL_PCTS]
    # The panel labels are drawn in data coordinates, so the figure has to grow
    # with the font size or they collide.
    fig, axes = plt.subplots(2, 2, figsize=(126, 52), squeeze=False,
                             gridspec_kw={"hspace": 0.42, "wspace": 0.55})
    rows = []
    for ax, (metric, name, label_pct) in zip(axes.ravel(), combos):
        table = _table(cells, label_pct, metric)
        average, p = _panel(ax, table, f"{name} — {label_pct}% labels")
        for method, rank in average.items():
            rows.append({"metric": metric, "label_pct": label_pct,
                         "method": method, "average_rank": rank,
                         "friedman_p": p, "n_streams": len(table)})
    fig.subplots_adjust(top=0.94)
    save(fig, "figure04_critical_difference", pad_inches=0.2)

    ranks = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(BENCHMARK_CSV), "average_ranks.csv")
    ranks.to_csv(out, index=False)
    print(f"  average ranks -> {os.path.relpath(out)}")
    for (metric, label_pct), group in ranks.groupby(["metric", "label_pct"]):
        best = group.loc[group.average_rank.idxmin()]
        print(f"  {metric:11s} {label_pct}% labels: best rank "
              f"{best.average_rank:.2f} ({best.method}), "
              f"Friedman p={best.friedman_p:.1e}, {int(best.n_streams)} streams")


if __name__ == "__main__":
    main()
