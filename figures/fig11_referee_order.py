"""Figure 11: high-order against binary relevance competence maps.

Per-stream difference in accuracy when the referee models dependencies between
members instead of treating them independently, shown for selection only (DES),
self-training only (SSL) and both together. Bar colour identifies the stream,
matching Figure 7; the sign is in the value and in the bar direction.

Reads: results/referee_ablation/runs.csv
"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from _common import (INK, MUTED, REFEREE_CSV, STREAM_COLOR, STREAM_LABEL,
                     save, style_axes)

MECHANISM = {"config_2": "DES", "config_13": "SSL", "config_12": "Both"}


def load():
    d = pd.read_csv(REFEREE_CSV)
    return d[d.global_acc.notna()]


def _delta(d, cfg, lp):
    """Paired per-stream difference in accuracy, in percentage points."""
    s = d[(d.config == cfg) & (d.label_pct == lp)]
    if s.empty:
        return pd.Series(dtype=float)
    p = s.pivot_table(index="dataset", columns="mode", values="global_acc")
    if "mlhat" not in p or "binary_relevance" not in p:
        return pd.Series(dtype=float)
    return ((p["mlhat"] - p["binary_relevance"]) * 100).dropna()


def fig_row_per_config(d, lp=5, cfgs=("config_2", "config_13", "config_12")):
    """The three mechanisms side by side, sharing one y axis."""
    ds_names = sorted(d.dataset.unique())
    labels = [("\n" * (i % 3) + STREAM_LABEL.get(n, n)) for i, n in enumerate(ds_names)]
    fig, axes = plt.subplots(1, len(cfgs), figsize=(11 * len(cfgs), 10.5),
                             sharey=True)
    lo_all, hi_all = [], []
    mid = len(cfgs) // 2
    for ci, (ax, cfg) in enumerate(zip(np.atleast_1d(axes), cfgs)):
        dd = _delta(d, cfg, lp).reindex(ds_names)
        n = dd.notna().sum()
        m = dd.dropna().mean()
        half = (stats.t.ppf(0.975, n - 1) * dd.dropna().std(ddof=1) / np.sqrt(n)
                ) if n > 1 else 0.0
        x = np.arange(len(ds_names))
        cols = [STREAM_COLOR[n] for n in ds_names]
        bars = ax.bar(x, np.nan_to_num(dd.values), 0.66, color=cols,
                      edgecolor="white", lw=1.0, zorder=2)
        for b, v, col in zip(bars, dd.values, cols):
            if not np.isfinite(v):
                continue
            ax.text(b.get_x()+b.get_width()/2, v + (0.08 if v >= 0 else -0.08),
                    f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top",
                    rotation=90, fontsize=48, fontweight="bold",
                    color=col, zorder=4)
        ax.axhline(0, color=INK, lw=3.0, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([""] * len(x))
        ax.tick_params(axis="x", length=0)
        ax.text(0.02, 0.97, MECHANISM.get(cfg, cfg), transform=ax.transAxes,
                ha="left", va="top", fontsize=76, fontweight="bold", color=INK)
        ax.text(0.985, 0.965, f"mean\n{m:+.2f} ± {half:.2f} pp",
                transform=ax.transAxes, ha="right", va="top", fontsize=40,
                fontweight="bold", color=INK, ma="center",
                bbox=dict(fc="white", ec=MUTED, lw=2.0, alpha=0.95,
                          boxstyle="round,pad=0.35"))
        style_axes(ax, labelsize=48, grid_axis='y')
        lo_all.append(np.nanmin(dd.values)); hi_all.append(np.nanmax(dd.values))
    span = max(hi_all) - min(lo_all)
    np.atleast_1d(axes)[0].set_ylim(min(lo_all) - 0.68*span,
                                    max(hi_all) + 0.80*span)
    np.atleast_1d(axes)[0].set_ylabel(
        "Δ accuracy (pp)\nHigh-order −\nBinary relevance",
        fontsize=48, fontweight="bold", color=INK, labelpad=16)
    fig.tight_layout(w_pad=1.5)
    save(fig, "figure11_referee_order")


if __name__ == "__main__":
    fig_row_per_config(load(), lp=5)
