"""Figure 8: effect of the self-training signal, under each selection strategy.

Two panels, one per contrast: the disagreement-based weight against no
pseudo-labelling, and against the relevance-based weight. Within a panel the x
axis is the selection strategy in use. Unlike selection, self-training needs a
single estimate of pseudo-label reliability rather than a relevance and diversity
trade-off, which is what these two contrasts separate.

Reads: results/ablation/runs.csv and results/seeds/runs.csv
"""
import matplotlib.pyplot as plt

from _common import save
from _effects import (COMBOS, FIGSIZE, SELECTION_LABEL, _fit_labels, _panel,
                      diff, load)

CONTRASTS = [
    ("none", "figure08a_selftraining_vs_none"),
    ("C", "figure08b_selftraining_vs_relevance"),
]


def main():
    df = load()
    strategies = ["none", "sqrtAM", "mmr"]
    categories = [SELECTION_LABEL[i] for i in strategies]
    for reference, name in CONTRASTS:
        series = []
        for (metric, label_pct), label, colour in COMBOS:
            values = [diff(df, metric, label_pct, i, "w", i, reference)
                      for i in strategies]
            series.append(([v for v, _ in values], [e for _, e in values],
                           label, colour))
        fig, ax = plt.subplots(figsize=FIGSIZE)
        _fit_labels(fig, ax, _panel(ax, categories, series, None, None))
        fig.tight_layout()
        save(fig, name, pad_inches=0.02)


if __name__ == "__main__":
    main()
