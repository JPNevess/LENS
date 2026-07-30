"""Figure 7: how well each selection signal tracks a learner's real accuracy.

One point per ensemble member, averaged over the run: the value of the signal
against the accuracy that member actually achieved. The three panels share a
scale so they can be compared directly, and r is averaged over the streams.

Reads: results/figure_data/signal_calibration.csv
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _common import FIGURE_DATA, INK, MUTED, STREAM_COLOR, STREAM_LABEL, save

CSV = os.path.join(FIGURE_DATA, "signal_calibration.csv")

PANELS = [
    ("margin", "Predictive\nmargin (M)"),
    ("binary_relevance", "Binary\nrelevance (Â)"),
    ("high_order", "Global\nhigh-order (Â)"),
]


def _correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or x[ok].std() == 0 or y[ok].std() == 0:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def build(label_pct=5):
    df = pd.read_csv(CSV)
    df = df[df.label_pct == label_pct]

    panels = []
    for signal, title in PANELS:
        points, correlations = [], []
        for stream, group in df[df.signal == signal].groupby("dataset"):
            points.append((stream, group.signal_mean.values,
                           group.real_accuracy.values))
            correlations.append(_correlation(group.signal_mean,
                                             group.real_accuracy))
        panels.append((title, points,
                       float(np.nanmean(np.asarray(correlations, float)))))

    # A shared window makes the three panels comparable at a glance, and lets
    # the y tick labels appear once instead of three times.
    values = np.concatenate([np.concatenate([x, y])
                             for _t, pts, _r in panels for _s, x, y in pts])
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    pad = 0.04 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    fig, axes = plt.subplots(1, 3, figsize=(34, 16))
    for ax, (title, points, r_mean) in zip(axes, panels):
        for stream, x, y in points:
            ax.scatter(x, y, s=420, color=STREAM_COLOR[stream], alpha=0.85,
                       edgecolor="white", lw=1.2, zorder=3,
                       label=STREAM_LABEL.get(stream, stream))
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=3.4, ls=":", zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{title}\n(r = {r_mean:.2f})", fontsize=66,
                     fontweight="bold", color=INK, pad=18)
        ax.grid(alpha=0.25, lw=1.0, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK, labelsize=66, width=2.6, length=13)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    axes[0].set_ylabel("real accuracy", fontsize=80, fontweight="bold",
                       color=INK, labelpad=18)

    # The legend needs two rows: on one row it is wider than the figure. The
    # 0.345 reservation is the smallest that leaves a band taller than the x
    # label; below it the label does not fit, above it the panels start to
    # shrink.
    fig.tight_layout(rect=(0, 0.345, 1, 1), w_pad=2.0)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, fontsize=56, ncol=5, loc="lower center",
                        bbox_to_anchor=(0.5, 0.0), framealpha=0.95,
                        markerscale=2.4, columnspacing=1.4, handletextpad=0.4,
                        borderpad=0.6, labelspacing=0.5)
    for text in legend.get_texts():
        text.set_fontweight("bold")

    # The x label is placed by measurement: with an equal aspect the axes are
    # repositioned after tight_layout, so the reserved band is not where the
    # label ends up.
    fig.canvas.draw()
    to_figure = fig.transFigure.inverted()
    legend_top = to_figure.transform((0, legend.get_window_extent().y1))[1]
    ticks_bottom = min(to_figure.transform((0, t.get_window_extent().y0))[1]
                       for t in axes[1].get_xticklabels())
    box = axes[1].get_position()
    fig.text((box.x0 + box.x1) / 2, (legend_top + ticks_bottom) / 2, "signal",
             ha="center", va="center", fontsize=80, fontweight="bold", color=INK)

    save(fig, "figure07_signal_calibration", pad_inches=0.2)
    for title, _pts, r_mean in panels:
        print(f"  {title.replace(chr(10), ' '):28s} r = {r_mean:.2f}")


if __name__ == "__main__":
    build(5)
