"""Figure 6: effect of dynamic ensemble selection, under each self-training
strategy.

Two panels, one per contrast: maximal marginal relevance against a uniform vote,
and against relevance-only selection. Within a panel the x axis is the
self-training strategy in use, which shows that the selection gain does not
depend on it. Each panel has its own y scale because the two contrasts differ by
an order of magnitude.

Reads: results/ablation/runs.csv and results/seeds/runs.csv
"""
import matplotlib.pyplot as plt

from _common import save
from _effects import (COMBOS, FIGSIZE, TRAINING_LABEL, _fit_labels, _panel,
                      diff, load)

CONTRASTS = [
    ("none", "figure06a_selection_vs_uniform"),
    ("sqrtAM", "figure06b_selection_vs_relevance"),
]


def main():
    df = load()
    strategies = ["none", "C", "w"]
    categories = [TRAINING_LABEL[t] for t in strategies]
    for reference, name in CONTRASTS:
        series = []
        for (metric, label_pct), label, colour in COMBOS:
            values = [diff(df, metric, label_pct, "mmr", t, reference, t)
                      for t in strategies]
            series.append(([v for v, _ in values], [e for _, e in values],
                           label, colour))
        fig, ax = plt.subplots(figsize=FIGSIZE)
        _fit_labels(fig, ax, _panel(ax, categories, series, None, None))
        fig.tight_layout()
        save(fig, name, pad_inches=0.02)


if __name__ == "__main__":
    main()
