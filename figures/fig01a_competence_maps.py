"""Figure 1 (top): label-free diversity estimates against ground truth.

One point per pair of learners, pooled over streams. The ground truth is the
disagreement between their true correctness states; the estimate is what each
approach can compute without labels. Plain disagreement only sees differing
predictions, the binary relevance map models each learner independently, and the
high-order map models their joint correctness.

Reads: results/figure_data/diversity_pairs.csv
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _common import FIGURE_DATA, INK, MUTED, save

CSV = os.path.join(FIGURE_DATA, "diversity_pairs.csv")

PANELS = [
    ("plain", "Plain disagreement", "#C44E52"),
    ("binary_relevance", "Binary relevance map", "#DD8452"),
    ("high_order", "High-order map", "#4C72B0"),
]


def build(label_pct=5):
    df = pd.read_csv(CSV)
    df = df[df.label_pct == label_pct]

    fig, axes = plt.subplots(1, 3, figsize=(21, 8))
    for ax, (estimator, title, colour) in zip(axes, PANELS):
        sub = df[df.estimator == estimator]
        if sub.empty:
            continue
        estimate = sub.estimate.values
        truth = sub.ground_truth.values
        r = (np.corrcoef(estimate, truth)[0, 1]
             if estimate.std() > 0 and truth.std() > 0 else np.nan)
        mae = np.mean(np.abs(estimate - truth))

        hi = min(1.0, max(estimate.max(), truth.max()) + 0.03)
        ax.plot([0.0, hi], [0.0, hi], color=MUTED, lw=2.0, ls=":", zorder=1)
        ax.scatter(truth, estimate, s=14, color=colour, alpha=0.3,
                   edgecolor="none", zorder=2)
        ax.set_title(f"{title}\nr = {r:.3f}   MAE = {mae:.3f}", fontsize=28,
                     fontweight="bold", color=INK, pad=16)
        # The x label goes on the middle panel only; all three share the axis.
        if estimator == "binary_relevance":
            ax.set_xlabel("ground truth diversity", fontsize=30,
                          fontweight="bold", color=INK, labelpad=14)
        ax.set_xlim(0.0, hi)
        ax.set_ylim(0.0, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK, labelsize=24)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        print(f"  {title:24s} r = {r:.3f}   MAE = {mae:.3f}")

    axes[0].set_ylabel("label-free diversity estimate", fontsize=30,
                       fontweight="bold", color=INK, labelpad=14)
    fig.tight_layout(w_pad=2.5)
    save(fig, "figure01a_competence_maps", pad_inches=0.2)


if __name__ == "__main__":
    build(5)
