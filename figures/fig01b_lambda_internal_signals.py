"""Figure 1 (middle and bottom): internal signals around an abrupt drift.

Top row: the fraction of members that were correct, and the disagreement among
the pairs that were both correct, for three fixed values of the
relevance/diversity trade-off. Bottom row: the same quantities as differences
against the balanced setting, which is what shows that more diversity preserves
disagreement before the drift and reduces redundancy after it.

Curves are aligned on the true drift positions and averaged over drifts and
streams.

Reads: results/figure_data/window_signals.csv
"""
import numpy as np
import matplotlib.pyplot as plt

from _common import INK, MUTED, save
from _lambda import (ABRUPT, LAMBDA_COLOUR, LAMBDA_NAME, SHORT_NAME,
                     internal_aligned, load_windows)

CONFIG = "config_12"
TAGS = ("w0.50", "w0.75", "w0.95")
REFERENCE = "w0.75"
XLO, XHI = -5000, 10000
DRIFT = "#C44E52"


def _style(ax):
    ax.grid(alpha=0.28, lw=0.9, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=40)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
    ax.axvline(0, color=DRIFT, lw=3.0, alpha=0.9, zorder=2)


def _fit_ylim(ax, curves, pad=0.10):
    """Scale y using only the part of the curves left visible in x."""
    vals = []
    for g, m in curves:
        v = np.asarray(m, float)[(np.asarray(g) >= XLO) & (np.asarray(g) <= XHI)]
        vals += list(v[np.isfinite(v)])
    if not vals:
        return
    lo, hi = min(vals), max(vals)
    d = (hi - lo) or 1.0
    ax.set_ylim(lo - pad * d, hi + pad * d)


def build(lp=5):
    REF = REFERENCE
    windows = load_windows()
    PAIRS = [(REF, t, LAMBDA_COLOUR[t],
              f"{SHORT_NAME[REF]} − {SHORT_NAME[t]}   ({m})")
             for t, m in ((TAGS[0], "more MMR"), (TAGS[2], "less MMR"))]
    aligned = {t: internal_aligned(windows, CONFIG, lp, ABRUPT, t)
               for t in TAGS}
    if all(a[0][1] is None for a in aligned.values()):
        print("  no window signals found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(25, 15),
                             gridspec_kw={"hspace": 0.58, "wspace": 0.52})
    (axP, axD), (axPd, axDd) = axes

    cP, cD = [], []
    for tag in TAGS:
        (gP, mP), (gD, mD) = aligned[tag]
        if mP is None:
            continue
        axP.plot(gP, mP, color=LAMBDA_COLOUR[tag], lw=4.5, label=LAMBDA_NAME[tag], zorder=3)
        axD.plot(gD, mD, color=LAMBDA_COLOUR[tag], lw=4.5, label=LAMBDA_NAME[tag], zorder=3)
        cP.append((gP, mP)); cD.append((gD, mD))

    cPd, cDd = [], []
    for ta, tb, col, lab in PAIRS:
        (gPa, mPa), (gDa, mDa) = aligned[ta]
        (gPb, mPb), (gDb, mDb) = aligned[tb]
        if mPa is None or mPb is None:
            continue
        axPd.plot(gPa, mPa - mPb, color=col, lw=4.5, label=lab, zorder=3)
        axDd.plot(gDa, mDa - mDb, color=col, lw=4.5, label=lab, zorder=3)
        cPd.append((gPa, mPa - mPb)); cDd.append((gDa, mDa - mDb))

    ylabs = ["correct\nmembers (%)", "pairwise\ndisagreement",
             "Δ correct\nmembers (pp)", "Δ pairwise\ndisagreement"]
    for ax, yl, curves in zip((axP, axD, axPd, axDd), ylabs,
                              (cP, cD, cPd, cDd)):
        ax.set_ylabel(yl, fontsize=44, fontweight="bold", color=INK, labelpad=16)
        ax.set_xlim(XLO, XHI)
        ax.set_xticks([-5000, 0, 5000, 10000])
        _fit_ylim(ax, curves, pad=0.06)
        _style(ax)
    for ax in (axPd, axDd):
        ax.axhline(0, color=INK, lw=2.0, zorder=2)
        ax.set_xlabel("instances since drift", fontsize=48, fontweight="bold",
                      color=INK, labelpad=16)

    fig.subplots_adjust(left=0.085, right=0.99, top=0.905, bottom=0.075)
    pT = axP.get_position(); pB = axPd.get_position()
    for ax_src, ybase in ((axP, pT.y1 + 0.012), (axPd, pB.y1 + 0.012)):
        h_, l_ = ax_src.get_legend_handles_labels()
        lg = fig.legend(h_, l_, fontsize=36, ncol=len(l_), framealpha=0.95,
                        loc="lower center", bbox_to_anchor=(0.5, ybase),
                        borderpad=0.5, handlelength=1.5, columnspacing=1.6,
                        handletextpad=0.5)
        for t in lg.get_texts():
            t.set_fontweight("bold")

    for ax in (axP, axD, axPd, axDd):
        ax.text(0, 0.99, " drift", transform=ax.get_xaxis_transform(),
                color=DRIFT, fontsize=40, fontweight="bold", va="top", ha="left",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
    save(fig, "figure01b_lambda_internal_signals", pad_inches=0.05)


if __name__ == "__main__":
    build(5)
