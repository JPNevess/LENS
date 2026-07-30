"""Figure 9: calibration of the signals that admit a pseudo-label.

9a: how pure the admitted pseudo-labels are as the admission threshold rises, and
what fraction of instances survive it. A signal is useful for self-training only
if precision keeps improving while coverage stays usable.

9b: risk, that is one minus precision, for the two signals that exist under both
referees. It shows where each one wins: the independent map is better in the
comfortable range, the high-order map in the high-purity tail, which is the
regime self-training actually operates in.

Counts are summed over streams at each threshold, so precision and coverage are
computed on the pooled admissions.

Reads: results/figure_data/admission_counts.csv
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from _common import FIGURE_DATA, INK, MUTED, save

CSV = os.path.join(FIGURE_DATA, "admission_counts.csv")

# Admission threshold used by the method.
PHI = 0.9
TAU_LO, TAU_HI = 0.80, 1.00
MIN_ADMITTED = 200

SIGNALS = [
    ("M", r"$M$", "#4C72B0", "o"),
    ("c", r"$c_{\hat{L}}$", "#DD8452", "s"),
    ("A_hat", r"$\hat{A}$", "#55A868", "^"),
    ("C", r"$\sqrt{\hat{A}\cdot M}$", "#8172B2", "D"),
    ("w", r"$\sqrt{\bar{A}\cdot c_{\hat{L}}}$", "#C44E52", "v"),
]

# Signals collected under both referees, for the second figure.
REFEREE_PAIRS = [
    ("A_hat", "binary_relevance_probe", r"$\hat{A}$"),
    ("C", "binary_relevance", r"$\sqrt{\hat{A}\cdot M}$"),
]
HIGH_ORDER_COLOUR, BINARY_COLOUR = "#8172B2", "#64B5CD"
CROSSOVER = 0.95


def _curve(df, signal, referee, label_pct):
    """Threshold, precision and coverage, pooled over streams."""
    sub = df[(df.signal == signal) & (df.referee == referee)
             & (df.label_pct == label_pct)]
    if sub.empty:
        return None
    pooled = sub.groupby("threshold")[["n_total", "n_admitted", "n_correct"]].sum()
    pooled = pooled[(pooled.index >= TAU_LO) & (pooled.index <= TAU_HI)]
    pooled = pooled[pooled.n_admitted >= MIN_ADMITTED]
    if pooled.empty:
        return None
    return (pooled.index.values,
            pooled.n_correct.values / pooled.n_admitted.values,
            pooled.n_admitted.values / pooled.n_total.values)


def _style(ax, labelsize=32):
    ax.grid(alpha=0.25, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK, labelsize=labelsize)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def precision_and_coverage(df, label_pct=5):
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(15, 15), sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1], "hspace": 0.10})

    for signal, label, colour, marker in SIGNALS:
        curve = _curve(df, signal, "high_order", label_pct)
        if curve is None:
            continue
        thresholds, precision, coverage = curve
        top.plot(thresholds, precision, "-", marker=marker, color=colour,
                 lw=5.0, ms=14, markevery=4, zorder=3, label=label)
        bottom.plot(thresholds, coverage, "-", marker=marker, color=colour,
                    lw=5.0, ms=14, markevery=4, zorder=3)

    for ax in (top, bottom):
        ax.axvline(PHI, color=MUTED, lw=2.6, ls="--", zorder=1)
        ax.set_xlim(TAU_LO, TAU_HI)
        ax.set_xticks([0.80, 0.85, 0.90, 0.95, 1.00])
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        _style(ax)
    top.text(PHI, top.get_ylim()[1], f"  Φ = {PHI}", color=MUTED, fontsize=32,
             fontweight="bold", va="top", ha="left")
    top.set_ylabel("precision of admitted\npseudo-labels", fontsize=34,
                   fontweight="bold", color=INK, labelpad=14)
    bottom.set_yscale("log")
    bottom.set_ylabel("coverage\nP(score ≥ t) [log]", fontsize=34,
                      fontweight="bold", color=INK, labelpad=14)
    bottom.set_xlabel("admission threshold t", fontsize=36, fontweight="bold",
                      color=INK, labelpad=12)
    # In the coverage panel: the precision curves span the full width of the
    # upper one and a legend there would sit on top of them.
    handles, labels = top.get_legend_handles_labels()
    legend = bottom.legend(handles, labels, fontsize=36, ncol=2,
                           framealpha=0.95, loc="lower left",
                           columnspacing=1.2, handlelength=1.6)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.tight_layout()
    save(fig, "figure09a_precision_vs_coverage", pad_inches=0.05)


def referee_effect(df, label_pct=5):
    fig, axes = plt.subplots(2, 1, figsize=(15, 22), sharex=True,
                             gridspec_kw={"hspace": 0.42})
    for ax, (signal, binary_referee, label) in zip(axes, REFEREE_PAIRS):
        high = _curve(df, signal, "high_order", label_pct)
        binary = _curve(df, signal, binary_referee, label_pct)
        if high is None or binary is None:
            continue
        ax.axvspan(TAU_LO, CROSSOVER, color=BINARY_COLOUR, alpha=0.10, zorder=0)
        ax.axvspan(CROSSOVER, TAU_HI, color=HIGH_ORDER_COLOUR, alpha=0.10,
                   zorder=0)
        ax.plot(binary[0], 1.0 - binary[1], color=BINARY_COLOUR, lw=6.0,
                zorder=3)
        ax.plot(high[0], 1.0 - high[1], color=HIGH_ORDER_COLOUR, lw=6.0,
                zorder=3)
        ax.set_ylabel(label, fontsize=54, fontweight="bold", color=INK,
                      labelpad=16)
        ax.set_xlim(TAU_LO, TAU_HI)
        ax.set_xticks([0.80, 0.85, 0.90, 0.95, 1.00])
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        _style(ax, labelsize=44)

    axes[1].set_xlabel("admission threshold", fontsize=48, fontweight="bold",
                       color=INK, labelpad=14)
    fig.text(0.02, 0.5, "risk = 1 − precision of admitted pseudo-labels",
             rotation=90, va="center", ha="left", fontsize=44,
             fontweight="bold", color=INK)
    # The two phrases sit once, between the panels, and stand in for a colour
    # legend: the colour of each phrase is the colour of its curve.
    middle = (axes[0].get_position().y0 + axes[1].get_position().y1) / 2
    fig.text(0.33, middle, "Binary relevance\nbetter", ha="center", va="center",
             fontsize=46, fontweight="bold", color=BINARY_COLOUR)
    fig.text(0.72, middle, "High-order\nbetter", ha="center", va="center",
             fontsize=46, fontweight="bold", color=HIGH_ORDER_COLOUR)
    save(fig, "figure09b_referee_effect", pad_inches=0.05)


def main(label_pct=5):
    df = pd.read_csv(CSV)
    precision_and_coverage(df, label_pct)
    referee_effect(df, label_pct)


if __name__ == "__main__":
    main(5)
