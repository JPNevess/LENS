"""Figures 10a and 10c: behaviour of the relevance/diversity trade-off around
drift, for gradual (left) and abrupt (right) changes.

10a is the diversity among the members that were correct, which rises after a
drift and is what motivates detecting one at all. 10c is the recovery in rolling
accuracy, and the difference between the dynamic policy and each fixed value.

Curves are aligned on the true drift positions and averaged over drifts and
streams. The two figures are written separately because they are placed
separately in the paper.

Reads: results/figure_data/window_signals.csv and results/lambda_study/history
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _common import INK, MUTED, save
from _lambda import (ABRUPT, GRADUAL, LAMBDA_COLOUR, accuracy_aligned,
                     internal_aligned, load_windows, smooth)

CONFIG = "config_12"
TAGS = ("w0.50", "w0.75", "w0.95")
PAIRS = [("w0.50", "#4C72B0", "dyn. − 0.50"),
         ("w0.75", "#55A868", "dyn. − 0.75"),
         ("w0.95", "#C44E52", "dyn. − 0.95")]
XLO, XHI = -2500, 5000
DRIFT = "#C44E52"
ADAPT = "#222222"

YLABELS = ["Div.\n(correct\ngroup)", "roll. acc.\n(%)", "roll. acc.\nΔ"]
PREFIX = {0: "λ =", 1: "λ =", 2: None}
PANEL = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
DRIFT_LABEL = [" gradual\n drift", " abrupt\n drift"]


import numpy as np


def _style(ax):
    ax.grid(alpha=0.28, lw=0.9, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=50, width=3.0, length=14)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
    ax.axvline(0, color=DRIFT, lw=3.2, alpha=0.9, zorder=2)


def _visible(g, y):
    g = np.asarray(g, float); y = np.asarray(y, float)
    m = (g >= XLO) & (g <= XHI) & np.isfinite(y)
    return y[m]


def _row_ylim(axes, curves, pad=0.08):
    """Same y scale on both panels of a row: the gradual and abrupt\npair is only comparable if it shares an axis."""
    vals = np.concatenate([_visible(g, y) for g, y in curves]) if curves else []
    if not len(vals):
        return
    lo, hi = float(vals.min()), float(vals.max())
    d = (hi - lo) or 1.0
    for ax in axes:
        ax.set_ylim(lo - pad * d, hi + pad * d)


def _draw_row(r, ax, dss, lp, ma, windows):
    """Draw one row of the study into ``ax`` and return its curves, so\nthe row can share a y scale afterwards."""
    cur = []
    if r == 0:
        for tag in TAGS:
            (_gP, _mP), (gD, mD) = internal_aligned(windows, CONFIG, lp, dss, tag)
            if mD is None:
                continue
            ax.plot(gD, mD, color=LAMBDA_COLOUR[tag], lw=5.0, label=tag[1:], zorder=3)
            cur.append((gD, mD))
    elif r == 1:
        for tag in TAGS:
            g, m = accuracy_aligned(windows, CONFIG, lp, dss, tag)
            if m is None:
                continue
            ax.plot(g, m, color=LAMBDA_COLOUR[tag], lw=5.0, label=tag[1:], zorder=3)
            cur.append((g, m))
        g, m = accuracy_aligned(windows, CONFIG, lp, dss, "adapt")
        if m is not None:
            ax.plot(g, m, color=ADAPT, lw=4.8, ls="--", label="dynamic",
                    zorder=4)
            cur.append((g, m))
    else:
        _ga, ma_ = accuracy_aligned(windows, CONFIG, lp, dss, "adapt")
        for tag, col_, lab in PAIRS:
            gb, mb = accuracy_aligned(windows, CONFIG, lp, dss, tag)
            if ma_ is None or mb is None:
                continue
            d = smooth(ma_ - mb, k=ma) if ma > 1 else (ma_ - mb)
            ax.plot(gb, d, color=col_, lw=5.2, ls="--", label=lab,
                    zorder=3)
            cur.append((gb, d))
    return cur


def build(lp, ma, rows, suffix, figsize):
    """One figure with the requested rows."""
    groups = [("gradual", GRADUAL), ("abrupt", ABRUPT)]

    windows = load_windows()

    fig, axes = plt.subplots(len(rows), 2, figsize=figsize, squeeze=False)
    cur = {r: [] for r in rows}
    for col, (_gname, dss) in enumerate(groups):
        for i, r in enumerate(rows):
            cur[r] += _draw_row(r, axes[i, col], dss, lp, ma, windows)

    last = len(rows) - 1
    for i, r in enumerate(rows):
        _row_ylim(axes[i], cur[r])
        for c in range(2):
            ax = axes[i, c]
            ax.set_xlim(XLO, XHI)
            ax.set_xticks([-2500, 0, 2500, 5000])
            _style(ax)
            if r == 2:
                ax.axhline(0, color=INK, lw=2.0, zorder=2)
            if c == 0:
                ax.set_ylabel(YLABELS[r], fontsize=56, fontweight="bold",
                              color=INK, labelpad=14)
    for col in (0, 1):
        axes[0, col].set_title(" ", fontsize=56, pad=145)

    fig.tight_layout(w_pad=3.0, h_pad=15.0)

    fig.canvas.draw()
    for i, r in enumerate(rows):
        h, l = axes[i, 0].get_legend_handles_labels()
        y1 = max(axes[i, c].get_position().y1 for c in (0, 1))
        pre = PREFIX[r]
        hh = ([Line2D([], [], ls="", marker="")] + h) if pre else h
        ll = ([pre] + l) if pre else l
        lg = fig.legend(hh, ll, fontsize=54, ncol=len(ll),
                        framealpha=0.95, loc="lower center",
                        bbox_to_anchor=(0.5, y1 + 0.007), borderpad=0.4,
                        labelspacing=0.3, handlelength=1.4,
                        columnspacing=0.9, handletextpad=0.4)
        for t in lg.get_texts():
            t.set_fontweight("bold")

    for i, r in enumerate(rows):
        for c in range(2):
            ax = axes[i, c]
            ax.text(0.015, 0.975, PANEL[2 * r + c], transform=ax.transAxes,
                    ha="left", va="top", fontsize=68, fontweight="bold",
                    color=INK, zorder=9,
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=2.0))
            ax.text(0, 0.02, DRIFT_LABEL[c], transform=ax.get_xaxis_transform(),
                    color=DRIFT, fontsize=54, fontweight="bold", va="bottom",
                    ha="left", zorder=9, linespacing=0.95,
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))

    fig.canvas.draw()
    y0 = min(axes[last, c].get_position().y0 for c in (0, 1))
    tl = max((t.get_window_extent().height
              for t in axes[last, 0].get_xticklabels()), default=0.0)
    fig.text(0.5, y0 - tl / (fig.get_size_inches()[1] * fig.dpi) - 0.030,
             "instances since drift", ha="center", va="top", fontsize=68,
             fontweight="bold", color=INK)

    save(fig, suffix)


if __name__ == "__main__":
    build(5, 1, [0], "figure10a_correct_group_diversity", (23, 11.2))
    build(5, 1, [1, 2], "figure10c_lambda_recovery", (23, 22.5))
