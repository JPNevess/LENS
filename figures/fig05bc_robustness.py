"""Figures 5b and 5c: robustness diagnostics of the pairwise comparisons.

5b shows the distribution of Cohen's d, win rate and break point over every pair
of methods, with the threshold that counts as a failure marked. 5c aggregates
those into a fragility rate, comparing arbitrary pairs of ablations against pairs
involving the full method.

Reads: results/sota_metrics/
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _common import INK, MUTED, RESULTS, save

_MET = os.path.join(RESULTS, "sota_metrics")
CSV_MULTI = os.path.join(_MET, "sota_metrics_multiseed.csv")
CSV_SINGLE = os.path.join(_MET, "sota_metrics.csv")


import os
TAU_D, TAU_W, TAU_B = 0.2, 0.6, 0.2
LPS = (5, 1)
BLUE, ORANGE, GREEN, RED = "#4C72B0", "#DD8452", "#55A868", "#C44E52"
DMAX = 2.0
PROPOSED = "config_12"

METRICS = [
    ("cohens_d",        "fail_magnitude",   "Cohen's d",        TAU_D),
    ("win_rate",        "fail_consistency", "Win rate",         TAU_W),
    ("breakdown_point", "fail_stability",   "Break point",      TAU_B),
]


def _style(ax, labelsize=46):
    ax.grid(axis="y", alpha=0.25, lw=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=labelsize)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")


def _load():
    path = CSV_MULTI if os.path.exists(CSV_MULTI) else CSV_SINGLE
    print(f"  source: {os.path.basename(path)}")
    df = pd.read_csv(path)
    return df[(df.winner != "config_0") & (df.loser != "config_0")]


def fig_diagnostics(df):
    df = df[df.target == "global_acc"]
    fig, axes = plt.subplots(len(LPS), 3, figsize=(26, 12))
    for r, lp in enumerate(LPS):
        sub = df[df.label_pct == lp]
        for c, (col, failcol, name, thr) in enumerate(METRICS):
            ax = axes[r, c]
            is_cohen = (col == "cohens_d")
            v = sub[col].values.astype(float)
            if is_cohen:
                v = np.clip(np.nan_to_num(v, nan=DMAX, posinf=DMAX, neginf=0.0), 0, DMAX)
                bins = np.linspace(0, DMAX, 21)
            else:
                bins = np.linspace(0, 1, 21)
            ax.axvspan(0.0, thr, color=RED, alpha=0.08, zorder=0)
            ax.hist(v, bins=bins, color=BLUE, alpha=0.85, edgecolor="white",
                    linewidth=0.5, zorder=2)
            ax.axvline(thr, color=RED, lw=3.0, ls="--", zorder=3)
            ymax = ax.get_ylim()[1]
            ax.set_ylim(0, ymax * 1.30)
            ax.text(0.03, 0.95, f"τ={thr:.1f}", transform=ax.transAxes,
                    color=RED, fontsize=46, fontweight="bold",
                    va="top", ha="left",
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
            ax.text(0.97, 0.95, f"{100*float(sub[failcol].mean()):.0f}% fail",
                    transform=ax.transAxes, ha="right", va="top", fontsize=50,
                    fontweight="bold", color=RED,
                    bbox=dict(fc="white", ec=RED, lw=1.6, alpha=0.92,
                              boxstyle="round,pad=0.32"))
            if c == 0:
                ax.set_ylabel(f"{lp}% labels", fontsize=58, color=INK,
                              fontweight="bold", labelpad=12)
            if r == 0:
                ax.set_title(name, fontsize=60, fontweight="bold", color=INK, pad=18)
            _style(ax, labelsize=46)
    fig.tight_layout()
    save(fig, "figure05b_diagnostics")


def fig_fragility(df):
    groups = [(t, lp) for t in ("global_acc", "f1_score") for lp in LPS]
    TT = {"global_acc": "Acc", "f1_score": "F1"}
    glabels = [f"{TT[t]}\n{lp}%" for (t, lp) in groups]
    frag_all, frag_vs12, fmag, fcon, fsta = [], [], [], [], []
    for (t, lp) in groups:
        s = df[(df.target == t) & (df.label_pct == lp)]
        vs12 = s[(s.winner == PROPOSED) | (s.loser == PROPOSED)]
        frag_all.append(s.fragile.mean())
        frag_vs12.append(vs12.fragile.mean() if len(vs12) else np.nan)
        fmag.append(s.fail_magnitude.mean()); fcon.append(s.fail_consistency.mean())
        fsta.append(s.fail_stability.mean())

    x = np.arange(len(groups))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(22, 10))

    w = 0.42
    for xs, vals, col, lab in [(x - w/2, frag_all, BLUE, "All pairs (ablation)"),
                               (x + w/2, frag_vs12, ORANGE, "Pairs vs LENS")]:
        bars = axA.bar(xs, vals, w, color=col, label=lab, zorder=2)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                axA.text(b.get_x()+b.get_width()/2, v + 0.014, f"{v:.2f}",
                         ha="center", va="bottom", fontsize=34,
                         fontweight="bold", color=INK)
    axA.set_xticks(x); axA.set_xticklabels(glabels, fontsize=42,
                                          fontweight="bold")
    axA.set_ylabel("Fragility rate", fontsize=50, fontweight="bold",
                   color=INK, labelpad=12)
    axA.set_ylim(0, min(1.0, max(np.nanmax(frag_vs12), max(frag_all)) + 0.26))
    axA.set_title("(a) Fragility rate", fontsize=54, fontweight="bold",
                  color=INK, pad=16)
    lg = axA.legend(fontsize=36, framealpha=0.95, loc="upper left")
    for t in lg.get_texts():
        t.set_fontweight("bold")
    _style(axA, labelsize=42)

    w3 = 0.28
    specs = [(fmag, BLUE,   "",   "Cohen's d (≤ 0.2)"),
             (fcon, ORANGE, "//", "Win rate (≤ 0.6)"),
             (fsta, GREEN,  "..", "Break point (≤ 0.2)")]
    for i, (vals, col, hatch, lab) in enumerate(specs):
        bars = axB.bar(x + (i-1)*w3, vals, w3, color=col, hatch=hatch,
                       edgecolor="white", linewidth=0.9, label=lab, zorder=2)
        for b, v in zip(bars, vals):
            axB.text(b.get_x()+b.get_width()/2, v + 0.009, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=34, rotation=90,
                     fontweight="bold", color=INK)
    axB.set_xticks(x); axB.set_xticklabels(glabels, fontsize=42,
                                          fontweight="bold")
    axB.set_ylabel("Pairs failing", fontsize=50, fontweight="bold",
                   color=INK, labelpad=12)
    axB.set_ylim(0, max(0.45, max(max(fmag), max(fcon), max(fsta)) + 0.34))
    axB.set_title("(b) Failing criterion", fontsize=54,
                  fontweight="bold", color=INK, pad=16)
    lg = axB.legend(fontsize=36, framealpha=0.95, loc="upper right")
    for t in lg.get_texts():
        t.set_fontweight("bold")
    _style(axB, labelsize=42)

    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.165, top=0.88,
                        wspace=0.24)
    save(fig, "figure05c_fragility")


def main():
    df = _load()
    fig_diagnostics(df)
    fig_fragility(df)


if __name__ == "__main__":
    main()
