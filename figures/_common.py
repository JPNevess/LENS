"""Paths, palette and save helper shared by the figure scripts.

Every figure reads only from ``results/`` and writes to ``figures/output/``, so
the whole set can be regenerated without re-running any experiment.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(_ROOT, "results")
FIGURE_DATA = os.path.join(RESULTS, "figure_data")
OUT_DIR = os.path.join(_ROOT, "figures", "output")

# Tables the figures read.
BENCHMARK_CSV = os.path.join(RESULTS, "ablation", "runs.csv")
SEEDS_CSV = os.path.join(RESULTS, "seeds", "runs.csv")
REFEREE_CSV = os.path.join(RESULTS, "referee_ablation", "runs.csv")
DRIFT_CSV = os.path.join(RESULTS, "drift_detection", "metrics.csv")
LAMBDA_HISTORY = os.path.join(RESULTS, "lambda_study", "history")

INK = "#222222"
MUTED = "#666666"

# The four series that recur across the ablation figures.
SERIES = [
    (("global_acc", 5), "Accuracy 5%", "#4C72B0"),
    (("global_acc", 1), "Accuracy 1%", "#8FB4D9"),
    (("f1_score", 5), "F1 5%", "#DD8452"),
    (("f1_score", 1), "F1 1%", "#F0B98E"),
]

# Streams, in the order used across the paper, with the short labels that fit
# on an axis.
STREAMS = ("AGR_a", "AGR_g", "CovtFD", "Electricity", "LED_a", "LED_g",
           "RBF_f", "RBF_m", "airlines")
STREAM_LABEL = {"CovtFD": "COVT", "Electricity": "ELEC", "airlines": "AIRL"}
STREAM_COLOR = {name: plt.get_cmap("tab10")(i % 10)
                for i, name in enumerate(STREAMS)}

# Streams excluded from the paper's tables.
EXCLUDED_STREAMS = ("Rialto", "NOAA", "airlines_without_AirportToFrom",
                    "RBF_a", "ForestCoverType", "PokerHand")


def save(fig, name, pad_inches=0.05):
    """Write a figure as PNG and PDF under ``figures/output``."""
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.join(OUT_DIR, name)
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight",
                    pad_inches=pad_inches)
    plt.close(fig)
    print(f"  {name}.png/.pdf")
    return stem


def style_axes(ax, labelsize=40, grid_axis="both"):
    """Common axis styling: no top/right spine, bold ticks, light grid."""
    if grid_axis != "none":
        ax.grid(axis=grid_axis, alpha=0.28, lw=0.9, zorder=0)
        ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK, labelsize=labelsize)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
