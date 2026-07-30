"""Shared machinery for the two component-effect figures (6 and 8).

Both isolate one component by taking the paired per-stream difference between the
two cells of the factorial grid that differ only in that component. Values are
drawn upright next to each bar, and the axis is opened by measuring the labels
after they are drawn rather than by reserving a guessed margin.

Reads: results/ablation/runs.csv and results/seeds/runs.csv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _common import (BENCHMARK_CSV, EXCLUDED_STREAMS, INK, SEEDS_CSV, SERIES,
                     save)

FIGSIZE = (13, 10)

# Cell of the factorial grid for each (selection, self-training) pair.
GRID = {
    ("none", "none"): "config_base", ("none", "C"): "config_7",
    ("none", "w"): "config_8",
    ("sqrtAM", "none"): "config_3", ("sqrtAM", "C"): "config_9",
    ("sqrtAM", "w"): "config_11",
    ("mmr", "none"): "config_4", ("mmr", "C"): "config_10",
    ("mmr", "w"): "config_12",
}
SELECTION_LABEL = {"none": "None", "sqrtAM": "\u221a(A\u00b7M)", "mmr": "MMR"}
TRAINING_LABEL = {"none": "None", "C": "\u221a(A\u00b7M)", "w": "\u221a(\u0100\u00b7c)"}
COMBOS = SERIES


def load():
    """Per-stream cell means, averaging the seeds of each cell.\n\nThe difference and its uncertainty are computed over these means, so\nthe unit of analysis is the stream and not the individual run."""
    key = ["dataset", "config", "label_pct", "seed"]
    cols = key + ["global_acc", "f1_score"]
    a = pd.read_csv(BENCHMARK_CSV)
    a = a[a.global_acc.notna() & ~a.dataset.isin(EXCLUDED_STREAMS)]
    if True:
        inc = pd.read_csv(SEEDS_CSV)
        inc = inc[inc.global_acc.notna() & ~inc.dataset.isin(EXCLUDED_STREAMS)]
        m = pd.concat([inc[cols], a[cols]]).drop_duplicates(subset=key, keep="first")
    else:
        m = a[cols]
    n = m.groupby(["dataset", "config", "label_pct"]).size()
    print(f"  {len(m)} runs over {len(n)} cells "
          f"(seeds per cell: min {n.min()}, median {int(n.median())}, max {n.max()})")
    return m.groupby(["dataset", "config", "label_pct"], as_index=False)[
        ["global_acc", "f1_score"]].mean()


def _cells(df, metric, lp, inf, tr):
    """Series of one grid cell, indexed by stream."""
    c = GRID.get((inf, tr))
    sub = df[(df.label_pct == lp) & (df.config == c)]
    return 100.0 * sub.set_index("dataset")[metric]


def diff(df, metric, lp, inf_a, tr_a, inf_b, tr_b):
    """Paired per-stream difference between two cells: mean and standard\nerror. Pairing matters because between-stream variation is large and\ncancels when the same stream is subtracted from itself."""
    A = _cells(df, metric, lp, inf_a, tr_a)
    B = _cells(df, metric, lp, inf_b, tr_b)
    d = (A - B).dropna()
    if not len(d):
        return np.nan, np.nan
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    return float(d.mean()), float(se)


def _panel(ax, cats, series, title, xlabel, fs_val=44):
    """Four series of bars per category, with upright value labels."""
    x = np.arange(len(cats)); w = 0.2
    lo, hi = 0.0, 0.0
    for vals, errs, _lab, _col in series:
        e0 = np.nan_to_num(np.asarray(errs, float))
        v0 = np.asarray(vals, float)
        lo = min(lo, np.nanmin(v0 - e0)); hi = max(hi, np.nanmax(v0 + e0))
    span = hi - lo

    texts = []
    for k, (vals, errs, lab, col) in enumerate(series):
        vals = np.asarray(vals, float); errs = np.asarray(errs, float)
        bars = ax.bar(x + (k - 1.5) * w, vals, w, color=col, edgecolor="black",
                      lw=0.9, label=lab, zorder=3,
                      yerr=np.nan_to_num(errs), capsize=11,
                      error_kw={"lw": 3.0, "zorder": 6, "ecolor": "#222222"})
        for b, v, e in zip(bars, vals, errs):
            if not np.isfinite(v):
                b.set_visible(False); continue
            bx = b.get_x() + b.get_width() / 2
            e = 0.0 if not np.isfinite(e) else e
            texts.append(ax.text(
                bx, v + e + 0.02 * span, f"{v:+.2f}",
                ha="center", va="bottom", rotation=90,
                fontsize=fs_val, fontweight="bold", color=col, zorder=8))
    ax.set_ylim(lo - (0.03 * span if lo < 0 else 0.01 * span), hi + 0.03 * span)
    ax.axhline(0, color="black", lw=1.6, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=58, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=52, fontweight="bold", labelpad=12)
    ax.set_ylabel("Δ score (pp)", fontsize=56, fontweight="bold", color=INK,
                  labelpad=14)
    if title:
        ax.set_title(title, fontsize=56, fontweight="bold", color=INK, pad=18)
    ax.tick_params(axis="y", labelsize=52, width=2.4, length=11)
    for t in ax.get_yticklabels():
        t.set_fontweight("bold")
    ax.tick_params(axis="x", length=0, pad=10)
    ax.grid(axis="y", alpha=0.3, lw=0.9, zorder=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return texts


def _fit_labels(fig, ax, texts, margin=0.015, passes=8):
    """Open the y axis until every upright value label fits.\n\nThe labels are measured after being drawn: their height is fixed in\ninches while the axis range is not, so any hand-picked margin breaks\nas soon as the font size or the figure changes."""
    for _ in range(passes):
        fig.canvas.draw()
        inv = ax.transData.inverted()
        y0, y1 = ax.get_ylim()
        lo, hi = y0, y1
        for t in texts:
            bb = t.get_window_extent()
            lo = min(lo, inv.transform((bb.x0, bb.y0))[1])
            hi = max(hi, inv.transform((bb.x1, bb.y1))[1])
        if lo >= y0 and hi <= y1:
            break
        d = hi - lo
        ax.set_ylim(lo - margin * d, hi + margin * d)
    return ax.get_ylim()


if __name__ == "__main__":
    main()
