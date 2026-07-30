"""Figure 10d: recall against precision for every drift detector.

Each point is one detector at its tuned ADWIN delta, averaged over the four
streams with known drift positions; bars are one standard error. Colour is the
strategy, marker shape the label rate, and a hollow marker means the binary
relevance variant of the referee. The inset magnifies the low-precision cluster,
where the points overlap in the full view.

Reads: results/drift_detection/metrics.csv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _common import DRIFT_CSV, INK, save

# One detector is a (signal, label rate) pair; colour encodes the strategy and
# the marker encodes the label rate.
BASE_DETECTORS = [
    ("sup",            "Supervised",             "supervised"),
    ("plain_dis",      "Plain-Disagreement",     "plain-disagreement"),
    ("studd_mlhat",    "Student-teacher",        "STUDD"),
    ("studd_br",       "Student-teacher (BR)",   "STUDD"),
    ("meta_acc_mlhat", "Meta-Supervised",        "meta-supervised"),
    ("meta_acc_br",    "Meta-Supervised (BR)",   "meta-supervised"),
    ("meta_dis_mlhat", "Meta-Disagreement",      "meta-disagreement"),
    ("meta_dis_br",    "Meta-Disagreement (BR)", "meta-disagreement"),
]
DETECTORS = [(f"{key}_{lp}", f"{label} ({lp}%)", family)
             for lp in (5, 1) for key, label, family in BASE_DETECTORS]
DET_KEYS = [d[0] for d in DETECTORS]
DET_FAM = {k: f for k, _, f in DETECTORS}
FAM_COLOR = {"supervised": "#4C72B0", "plain-disagreement": "#55A868",
             "STUDD": "#937860", "meta-supervised": "#DD8452",
             "meta-disagreement": "#8172B2"}


MARK = {5: "o", 1: "^"}
def _is_br(key):
    return "_br_" in key


ZOOM_X, ZOOM_Y = (0.555, 0.905), (0.055, 0.228)
ZOOM_ISO_F1    = (0.15, 0.20, 0.25, 0.30)
ZOOM_X_INSET, ZOOM_Y_INSET = (0.648, 0.912), (0.062, 0.212)
ZOOM_ANN = {
    "meta_dis_mlhat_5":  ("Meta-Disagreement · 5%",      (-30,  0), "right", "center"),
    "meta_dis_mlhat_1":  ("Meta-Disagreement · 1%",      (-30,  0), "right", "center"),
    "plain_dis_1":       ("Plain-Disagreement · 1%",     ( 30,  4), "left",  "center"),
    "studd_mlhat_1":     ("Student-teacher (global) · 1%", ( 30, 0), "left", "center"),
    "meta_acc_mlhat_1":  ("Meta-Supervised (global) · 1%", ( 30, 0), "left", "center"),
    "studd_br_1":        ("Student-teacher\n(BR) · 1%",  (-26, 26), "right", "bottom"),
    "meta_acc_mlhat_5":  ("Meta-Supervised (global) · 5%", (-30, 0), "right", "center"),
}
ZOOM_ANN_INSET = {
    "meta_dis_mlhat_5":  ("Meta-Disagr. 5%",           (-24,  0), "right", "center"),
    "meta_dis_mlhat_1":  ("Meta-Disagr. 1%",           (-24,  0), "right", "center"),
    "plain_dis_1":       ("Plain-Disagreement",        ( 24, 11), "left",  "center"),
    "studd_mlhat_1":     ("Student-teacher (global)",  ( 24, -6), "left",  "center"),
    "meta_acc_mlhat_1":  ("Meta-Supervised (global)",  ( 24, -9), "left",  "center"),
    "studd_br_1":        ("Student-teacher\n(BR)",     (-22, 20), "right", "bottom"),
    "meta_acc_mlhat_5":  ("Meta-Superv.\n(global) 5%", (-24,  0), "right", "center"),
}


def _mean_se(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan
    return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(v.size))
                             if v.size > 1 else 0.0)


def _collect():
    """One row per detector: family, label rate, recall, precision, F1."""
    df = pd.read_csv(DRIFT_CSV)
    pts = []
    for k in DET_KEYS:
        g = df[df.detector == k]
        if g.empty:
            continue
        rm, rs = _mean_se(g["recall"]); pm, ps = _mean_se(g["precision"])
        f1, _ = _mean_se(g["F1"])
        lp = int(k.rsplit("_", 1)[1])
        pts.append((k, DET_FAM[k], lp, rm, rs, pm, ps, f1))
    return pts


def _marker(ax, k, fam, lp, rm, pm, ms, label=None):
    """Draw one detector; hollow marker means binary relevance."""
    col = FAM_COLOR[fam]
    if _is_br(k):
        ax.plot(rm, pm, MARK[lp], ms=ms, mfc="white", mec=col,
                mew=ms / 5.7, zorder=4, label=label)
    else:
        ax.plot(rm, pm, MARK[lp], ms=ms, color=col, mec="white",
                mew=ms / 10.0, zorder=4, label=label)
    return col


def main():
    pts = _collect()

    fig, ax = plt.subplots(figsize=(20, 20))
    gx = np.linspace(0.02, 1, 400)
    for f1 in (0.2, 0.4, 0.6, 0.8):
        gy = f1 * gx / np.clip(2 * gx - f1, 1e-6, None)
        m = (gy > 0) & (gy <= 1.02) & (gx > f1 / 2)
        ax.plot(gx[m], gy[m], color="#CCCCCC", lw=3.2, zorder=1)
        ax.text(1.004, f1 / np.clip(2 - f1, 1e-6, None), f"F1={f1:.1f}",
                fontsize=52, fontweight="bold", color="#999999", va="center")

    P = sorted([(r, p) for *_x, r, _rs, p, _ps, _f in pts], reverse=True)
    front, best = [], -1
    for r, p in P:
        if p > best:
            front.append((r, p)); best = p
    if len(front) > 1:
        fr = sorted(front)
        ax.plot([f[0] for f in fr], [f[1] for f in fr], ls="--", lw=4.0,
                color=INK, alpha=0.55, zorder=2)

    seen = set()
    for k, fam, lp, rm, rs, pm, ps, f1 in pts:
        col = FAM_COLOR[fam]
        ax.errorbar(rm, pm, xerr=rs, yerr=ps, fmt="none", ecolor=col,
                    elinewidth=3.6, capsize=13, capthick=3.6, alpha=0.75, zorder=3)
        _marker(ax, k, fam, lp, rm, pm, 54,
                label=dict((f, f) for f in FAM_COLOR)[fam] if fam not in seen else None)
        seen.add(fam)

    ax.set_xlabel("Recall", fontsize=78, fontweight="bold", color=INK, labelpad=18)
    ax.set_ylabel("Precision", fontsize=78, fontweight="bold", color=INK,
                  labelpad=18)
    ax.set_xlim(-0.03, 1.10); ax.set_ylim(-0.03, 1.60)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(alpha=0.25, lw=1.2, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=76, width=3.6, length=19)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")

    axins = ax.inset_axes([0.015, 0.555, 0.88, 0.425])
    _draw_zoom(axins, pts, {}, fs_ann=24, fs_tick=56, fs_iso=24, ms=36,
               boxed=True, iso_labels=False, xlim=ZOOM_X_INSET,
               ylim=ZOOM_Y_INSET, elw=3.4, capsize=12)
    axins.set_xticks([0.65, 0.75, 0.85])
    axins.set_yticks([0.08, 0.12, 0.16, 0.20])
    axins.set_yticklabels(["0.08", "0.12", "0.16", "0.20"])
    axins.yaxis.tick_right()
    axins.xaxis.tick_top()
    axins.tick_params(width=3.0, length=16)
    for t in axins.get_xticklabels() + axins.get_yticklabels():
        t.set_fontweight("bold")
    rect, lines = ax.indicate_inset_zoom(axins, edgecolor=INK, alpha=0.9,
                                         lw=3.4)
    for ln in lines:
        ln.set(lw=2.4, color=INK, alpha=0.45, ls="-", zorder=1)

    h, l = ax.get_legend_handles_labels()
    h += [Line2D([], [], ls="", marker=MARK[5], ms=50, color="#555555"),
          Line2D([], [], ls="", marker=MARK[1], ms=50, color="#555555"),
          Line2D([], [], ls="", marker="s", ms=50, color="#555555"),
          Line2D([], [], ls="", marker="s", ms=50, mfc="white", mec="#555555",
                 mew=8.0)]
    l += ["5% labels", "1% labels", "Global high-order", "Binary relevance"]

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.375)
    leg = fig.legend(h, l, fontsize=54, ncol=2, loc="upper center",
                     bbox_to_anchor=(0.5, 0.275), framealpha=0.0,
                     borderpad=0.4, labelspacing=0.55, handletextpad=0.6,
                     handlelength=1.3, columnspacing=2.0)
    for t in leg.get_texts():
        t.set_fontweight("bold")

    save(fig, "figure10d_recall_vs_precision")
    print(f"  {len(pts)} detectors, best F1: {max(pts, key=lambda t: t[-1])[0]}")


def _draw_zoom(ax, pts, ann, fs_ann, fs_tick, fs_iso, ms, boxed=False,
               iso_labels=True, iso_side="right", xlim=None, ylim=None,
               elw=2.6, capsize=9):
    """Draw the magnified window into ``ax``."""
    x0, x1 = xlim or ZOOM_X; y0, y1 = ylim or ZOOM_Y
    inside = [p for p in pts if x0 <= p[3] <= x1 and y0 <= p[5] <= y1]

    gx = np.linspace(x0, x1, 400)
    xe = x1 if iso_side == "right" else x0
    for f1 in ZOOM_ISO_F1:
        gy = f1 * gx / np.clip(2 * gx - f1, 1e-6, None)
        ax.plot(gx, gy, color="#CCCCCC", lw=2.4, zorder=1)
        ye = f1 * xe / (2 * xe - f1)
        if iso_labels and y0 + 0.004 < ye < y1 - 0.004:
            ax.text(xe + (-0.002 if iso_side == "right" else 0.002), ye,
                    f"F1={f1:.2f}", fontsize=fs_iso, fontweight="bold",
                    color="#8C8C8C", va="bottom", ha=iso_side, zorder=2)

    for k, fam, lp, rm, rs, pm, ps, f1 in inside:
        col = FAM_COLOR[fam]
        ax.errorbar(rm, pm, xerr=rs, yerr=ps, fmt="none", ecolor=col,
                    elinewidth=elw, capsize=capsize, capthick=elw, alpha=0.55,
                    zorder=3)
        _marker(ax, k, fam, lp, rm, pm, ms)
        if k in ann:
            txt, (dx, dy), ha, va = ann[k]
            ax.annotate(txt, (rm, pm), textcoords="offset points",
                        xytext=(dx, dy), ha=ha, va=va, fontsize=fs_ann,
                        fontweight="bold", color=col, zorder=6,
                        bbox=dict(fc="white", ec="none", alpha=0.72, pad=1.5))

    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.grid(alpha=0.25, lw=1.0, zorder=0)
    if boxed:
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(INK); s.set_linewidth(3.0)
        ax.set_facecolor("white")
    else:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=fs_tick)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
    return inside


def _shape_legend(ax, fs, loc, ms=26):
    """Legend for marker shape and fill only; colour is in the labels."""
    h = [Line2D([], [], ls="", marker=MARK[5], ms=ms, color="#555555"),
         Line2D([], [], ls="", marker=MARK[1], ms=ms, color="#555555"),
         Line2D([], [], ls="", marker="s", ms=ms, color="#555555"),
         Line2D([], [], ls="", marker="s", ms=ms, mfc="white", mec="#555555",
                mew=5.0)]
    l = ["5% labels", "1% labels", "Global high-order", "Binary relevance"]
    leg = ax.legend(h, l, fontsize=fs, loc=loc, framealpha=0.95,
                    borderpad=0.6, labelspacing=0.4, handletextpad=0.5)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    return leg


def zoom():
    """Standalone version of the inset, for use outside the main figure."""
    pts = _collect()
    fig, ax = plt.subplots(figsize=(22, 14))
    inside = _draw_zoom(ax, pts, ZOOM_ANN, fs_ann=30, fs_tick=40, fs_iso=28,
                        ms=32)
    ax.set_xlabel("Recall", fontsize=52, fontweight="bold", color=INK, labelpad=16)
    ax.set_ylabel("Precision", fontsize=52, fontweight="bold", color=INK,
                  labelpad=16)
    _shape_legend(ax, 32, "lower left")

    fig.tight_layout()
    save(fig, "figure10d_recall_vs_precision_zoom")
    print(f"  {len(inside)}/{len(pts)} detectors inside the zoom window")


if __name__ == "__main__":
    main()
    zoom()
