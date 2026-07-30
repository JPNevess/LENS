"""Figure 5a: relative importance of each signal, by functional ANOVA.

The two panels are the two axes of the factorial design: how much of the variance
in the final score each signal explains when used for selection, and when used
for self-training. Bars are the four target/label-rate combinations.

Reads: results/component_importance/
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _common import INK, RESULTS, SERIES, save

_CI = os.path.join(RESULTS, "component_importance")
_MULTISEED = os.path.join(_CI, "component_importance_multiseed.csv")
CSV = _MULTISEED if os.path.exists(_MULTISEED) else os.path.join(
    _CI, "component_importance.csv")
COMBOS = SERIES


import os
GROUPS = [("(a) Inference", ["inf_margin", "inf_acc", "inf_div"]),
          ("(b) Training",  ["tr_margin", "tr_acc", "tr_disagree"])]
FEATURES = [f for _, fs in GROUPS for f in fs]
FLABEL   = {"inf_margin": "Margin", "inf_acc": "Relevance", "inf_div": "Diversity",
            "tr_margin": "Margin", "tr_acc": "Relevance", "tr_disagree": "Diversity"}


def main():
    df = pd.read_csv(CSV)
    t = df[df.table == "table4_everything"]

    fig, ax = plt.subplots(figsize=(24, 10.5))
    x = np.arange(len(FEATURES)); w = 0.2
    vmax = 0.0
    for k, ((tgt, lp), lab, col) in enumerate(COMBOS):
        sub = t[(t.target == tgt) & (t.label_pct == lp)].set_index("feature")
        vals = [100 * sub.loc[f, "fanova_importance"] for f in FEATURES]
        bars = ax.bar(x + (k - 1.5) * w, vals, w, color=col, edgecolor="black",
                      lw=0.8, label=lab, zorder=3)
        vmax = max(vmax, max(vals))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.0f}",
                    ha="center", va="bottom", rotation=0, fontsize=40,
                    fontweight="bold", color=col, zorder=4)

    ax.set_ylim(0, vmax * 1.55)
    ax.set_xlim(-0.62, len(FEATURES) - 0.38)

    n_inf = len(GROUPS[0][1])
    ax.axvspan(n_inf - 0.5, len(FEATURES) - 0.38, color="#000000", alpha=0.045,
               zorder=0)
    ax.axvline(n_inf - 0.5, color="#555555", lw=1.6, ls="-", zorder=2)
    tr = ax.get_xaxis_transform()
    for name, feats in GROUPS:
        centre = np.mean([FEATURES.index(f) for f in feats])
        ax.text(centre, 1.035, name, transform=tr, ha="center", va="bottom",
                fontsize=50, fontweight="bold", color=INK, clip_on=False)

    ax.set_xticks(x)
    ax.set_xticklabels([("\n" if i % 3 == 1 else "") + FLABEL[f]
                        for i, f in enumerate(FEATURES)],
                       fontsize=46, fontweight="bold")
    ax.tick_params(axis="y", labelsize=38)
    for t in ax.get_yticklabels():
        t.set_fontweight("bold")
    ax.tick_params(axis="x", length=0, pad=12)
    ax.set_ylabel("fANOVA importance (%)", fontsize=44, fontweight="bold",
                  labelpad=14)
    ax.grid(axis="y", alpha=0.3, lw=0.9, zorder=1)
    ax.set_axisbelow(True)
    leg = ax.legend(fontsize=40, ncol=4, framealpha=0.95, loc="upper center",
                    columnspacing=1.1, handlelength=1.5, borderpad=0.5)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    save(fig, "figure05a_signal_importance")


if __name__ == "__main__":
    main()
