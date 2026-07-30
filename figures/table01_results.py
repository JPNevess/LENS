"""Table 1: accuracy per stream and label rate, with the robustness summary.

Cells are mean and standard deviation over the seeds that ran. Below the streams:
the average, the mean rank over streams, and four pairwise diagnostics of every
method against the proposed one -- effect size, win rate, break point, and
whether the pair fails any of them.

Writes LaTeX to figures/output/table01_results.tex.

Reads: results/ablation/runs.csv and results/seeds/runs.csv
"""
import os

import numpy as np
import pandas as pd

from _benchmark import METHODS, PROPOSED, STREAMS, LABEL_PCTS, load
from _common import OUT_DIR

CONFIGS = [c for c, _ in METHODS]
COLS = [(label, config) for config, label in METHODS]
TAU_D, TAU_W, TAU_B = 0.2, 0.6, 0.2
LENS = PROPOSED
DATASETS = STREAMS
LPS = LABEL_PCTS


import os


def _cell(m, s, n, bold):
    if not np.isfinite(m):
        return "--"
    v = f"{100*m:.2f}"
    if n >= 2 and np.isfinite(s):
        v += rf"$_{{{100*s:.2f}}}$"
    if bold:
        v = rf"\textbf{{{v}}}"
    return v


def breakdown_point(delta):
    k = len(delta); s = float(np.sum(delta))
    if s <= 0:
        return 0.0
    for m, d in enumerate(sorted(delta, reverse=True), start=1):
        s -= d
        if s <= 0:
            return m / k
    return 1.0


def sota_rows(g, metric):
    """Pairwise diagnostics of every method against the proposed one.

    Effect size, win rate, break point, and whether the pair fails any of the
    three, computed per label rate with a fixed comparison direction.
    """
    out = {}
    for lp in LPS:
        sub = g[g.label_pct == lp]
        piv = sub.pivot_table(index="dataset", columns="config",
                              values=f"{metric}_m")
        for _, cfg in COLS:
            if cfg == LENS:
                continue
            both = piv[[LENS, cfg]].dropna()
            delta = (both[LENS] - both[cfg]).values
            sd = delta.std(ddof=1)
            d  = float(delta.mean() / sd) if sd > 0 else float("inf")
            wr = float((delta > 0).mean())
            bp = breakdown_point(delta)
            frag = (d <= TAU_D) or (wr <= TAU_W) or (bp <= TAU_B)
            out[(lp, cfg)] = (d, wr, bp, frag)
    return out


def build_table(g, metric, caption, label):
    mcol, scol = f"{metric}_m", f"{metric}_s"
    L = []
    L.append(r"\begin{table*}[t]")
    L.append(r"\centering")
    L.append(r"\setlength{\tabcolsep}{2pt}")
    L.append(r"\renewcommand{\arraystretch}{0.8}")
    L.append(rf"\caption{{{caption}}}")
    L.append(rf"\label{{{label}}}")
    L.append(r"\begin{tabular}{l|c|ccccc|cccc|cc}")
    L.append(r"\toprule")
    L.append(r"\textbf{Data} & \textbf{\%}")
    L.append("& " + " & ".join(h for h, _ in COLS[:5]))
    L.append("& " + " & ".join(h for h, _ in COLS[5:9]))
    L.append("& " + " & ".join(h for h, _ in COLS[9:]) + r" \\")
    L.append(r"\midrule")
    L.append("")

    means = {}
    for lab, ds in DATASETS:
        L.append(rf"\multirow{{2}}{{*}}{{{lab}}}")
        for lp in LPS:
            sub = g[(g.dataset == ds) & (g.label_pct == lp)].set_index("config")
            vals = {c: (sub.loc[c, mcol] if c in sub.index else np.nan)
                    for c in CONFIGS}
            best = max(vals, key=lambda c: (vals[c] if np.isfinite(vals[c]) else -1))
            cells = []
            for c in CONFIGS:
                m = vals[c]
                s = sub.loc[c, scol] if c in sub.index else np.nan
                n = int(sub.loc[c, "n"]) if c in sub.index else 0
                cells.append(_cell(m, s, n, c == best))
                means.setdefault((lp, c), []).append(m)
            L.append(f"& {lp} & " + " & ".join(cells) + r" \\")
        L.append("")
    L.append(r"\midrule")
    L.append("")

    L.append(r"\multirow{2}{*}{Avg $\uparrow$}")
    for lp in LPS:
        avgs = {c: np.nanmean(means[(lp, c)]) for c in CONFIGS}
        best = max(avgs, key=avgs.get)
        cells = [(rf"\textbf{{{100*avgs[c]:.2f}}}" if c == best
                  else f"{100*avgs[c]:.2f}") for c in CONFIGS]
        L.append(f"& {lp} & " + " & ".join(cells) + r" \\")
    L.append("")

    L.append(r"\multirow{2}{*}{Rank $\downarrow$}")
    for lp in LPS:
        M = np.array([means[(lp, c)] for c in CONFIGS])
        ranks = np.zeros_like(M)
        for j in range(M.shape[1]):
            order = pd.Series(M[:, j]).rank(ascending=False)
            ranks[:, j] = order.values
        rmean = ranks.mean(axis=1)
        best = int(np.argmin(rmean))
        cells = [(rf"\textbf{{{r:.2f}}}" if i == best else f"{r:.2f}")
                 for i, r in enumerate(rmean)]
        L.append(f"& {lp} & " + " & ".join(cells) + r" \\")
    L.append("")
    L.append(r"\midrule")
    L.append("")

    sr = sota_rows(g, metric)
    def _row(name, idx, fmt):
        lines = [rf"\multirow{{2}}{{*}}{{{name}}}"]
        for k, lp in enumerate(LPS):
            cells = [fmt(sr[(lp, c)][idx]) for _, c in COLS if c != LENS]
            if name.startswith("$d$") and k == 1:
                tail = (r" & \multirow{8}{*}[1em]{\rotatebox[origin=c]{-90}"
                        r"{\textbf{Pair vs. LENS}}} \\")
            else:
                tail = r" & \\"
            lines.append(f"& {lp} & " + " & ".join(cells) + tail)
        lines.append("")
        return lines
    L += _row(r"$d$ $\uparrow$",  0, lambda v: f"{v:.2f}")
    L += _row(r"$WR$ $\uparrow$", 1, lambda v: f"{v:.2f}")
    L += _row(r"$BP$ $\uparrow$", 2, lambda v: f"{v:.2f}")
    L += _row(r"$F$",             3, lambda v: ("Yes" if v else "No"))

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table*}")
    return "\n".join(L)


def main():
    cells = load()
    caption = (r"Accuracy as mean$_{std}$ (\%) over multiple runs under 1\% "
               r"and 5\% labelled data. Best result per stream in bold.")
    text = build_table(cells, "acc", caption, "tab:results")
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "table01_results.tex")
    with open(out, "w") as handle:
        handle.write(text + chr(10))
    print("  table01_results.tex")


if __name__ == "__main__":
    main()
