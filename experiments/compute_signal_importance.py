"""Variance decomposition of the contribution of each signal.

Fits the factorial design and reports how much of the variance in the final score
each signal explains, separately for selection and for self-training. Feeds
Figure 5a.

    python experiments/compute_signal_importance.py --multiseed
"""

import argparse
import glob
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import os, warnings, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUDE = ("Rialto", "NOAA", "airlines_without_AirportToFrom", "RBF_a", "ForestCoverType")

FEATURES = ["inf_margin", "inf_acc", "inf_div", "tr_margin", "tr_disagree", "tr_acc"]
FLABEL = {"inf_margin": "Inf:Margin", "inf_acc": "Inf:Acc", "inf_div": "Inf:Diversity",
          "tr_margin": "Tr:Margin", "tr_disagree": "Tr:Disagree", "tr_acc": "Tr:Acc"}
FCOLOR = {"inf_margin": "#4C72B0", "inf_acc": "#4C72B0", "inf_div": "#4C72B0",
          "tr_margin": "#DD8452", "tr_disagree": "#DD8452", "tr_acc": "#DD8452"}

DESIGN = {
    0:  (0,0,0, 0,0,0), 1:  (1,0,0, 0,0,0), 2:  (0,1,0, 0,0,0), 3:  (1,1,0, 0,0,0),
    4:  (1,1,1, 0,0,0), 5:  (0,0,0, 1,0,0), 6:  (0,0,0, 0,1,0), 7:  (0,0,0, 1,0,1),
    8:  (0,0,0, 0,1,1), 9:  (1,1,0, 1,0,1), 10: (1,1,1, 1,0,1), 11: (1,1,0, 0,1,1),
    12: (1,1,1, 0,1,1), 13: (0,0,0, 0,0,1),
}
def _cfg_name(i): return "config_base" if i == 0 else f"config_{i}"

TABLES = {
    "table4_everything": (list(range(14)),            FEATURES),
    "table5_inference":  ([0,1,2,3,4],                ["inf_margin","inf_acc","inf_div"]),
    "table6_training":   ([0,5,6,13,7,8],             ["tr_margin","tr_disagree","tr_acc"]),
}
TARGETS = [("global_acc", "Accuracy"), ("f1_score", "F1")]
LPS = [(5, "5%"), (1, "1%")]


def load(csv):
    df = pd.read_csv(csv)
    df = df[df.global_acc.notna() & ~df.dataset.isin(EXCLUDE)]
    return df


INCERTEZAS = os.path.join(_ROOT, "results", "seeds", "runs.csv")


def load_multiseed(csv):
    a = load(csv)
    key = ["dataset", "config", "label_pct", "seed"]
    cols = key + ["global_acc", "f1_score"]
    if os.path.exists(INCERTEZAS):
        inc = pd.read_csv(INCERTEZAS)
        inc = inc[inc.global_acc.notna() & ~inc.dataset.isin(EXCLUDE)]
        m = pd.concat([inc[cols], a[cols]]).drop_duplicates(subset=key, keep="first")
    else:
        print("  [note] seeds table not found, using the single-seed grid only.")
        m = a[cols]
    n = m.groupby(["dataset", "config", "label_pct"]).size()
    print(f"  multiseed: {len(m)} runs over {len(n)} cells "
          f"(seeds per cell: min {n.min()}, median {int(n.median())}, max {n.max()})")
    return m.groupby(["dataset", "config", "label_pct"], as_index=False)[
        ["global_acc", "f1_score"]].mean()


def build(df, cfg_nums, lp, target):
    """Long table: one row per (configuration, stream) with the indicators and the
    target, in percent.
    """
    rows = []
    sub = df[df.label_pct == lp]
    for i in cfg_nums:
        name = _cfg_name(i)
        d = sub[sub.config == name]
        des = dict(zip(FEATURES, DESIGN[i]))
        for _, r in d.iterrows():
            rows.append({**des, "dataset": r.dataset, "y": 100.0 * r[target]})
    return pd.DataFrame(rows)


def interaction_terms(data, feats):
    import itertools
    M = np.column_stack([np.ones(len(data))] +
                        [data[f].values.astype(float) for f in feats])
    rank = np.linalg.matrix_rank(M)
    chosen = []
    for a, b in itertools.combinations(feats, 2):
        col = (data[a].values * data[b].values).astype(float)
        M2 = np.column_stack([M, col])
        if np.linalg.matrix_rank(M2) > rank:
            M, rank = M2, rank + 1
            chosen.append(f"{a}:{b}")
    return chosen


def ols_coefs(data, feats, interactions=False):
    terms = list(feats)
    inter = interaction_terms(data, feats) if interactions else []
    formula = "y ~ " + " + ".join(terms + inter) + " + C(dataset)"
    res = smf.ols(formula, data=data).fit()
    out = {}
    for f in feats:
        ci = res.conf_int().loc[f]
        out[f] = {"coef": res.params[f], "lo": ci[0], "hi": ci[1], "p": res.pvalues[f]}
    return out, res.rsquared


def rf_importance(data, feats):
    imp = {f: [] for f in feats}
    for ds, sub in data.groupby("dataset"):
        if sub[feats].drop_duplicates().shape[0] < 2:
            continue
        X = sub[feats].values; y = sub["y"].values
        rf = RandomForestRegressor(n_estimators=400, random_state=42).fit(X, y)
        pi = permutation_importance(rf, X, y, n_repeats=30, random_state=42)
        raw = np.maximum(pi.importances_mean, 0); tot = raw.sum() + 1e-9
        for f, v in zip(feats, raw / tot):
            imp[f].append(v)
    return {f: float(np.mean(v)) if v else 0.0 for f, v in imp.items()}


def fig_coef(all_coefs, table, feats, out):
    """OLS coefficients: four bars per feature (accuracy/F1 at 5%/1%) with 95% CIs."""
    combos = [(t, lp, f"{tl} {sl}") for (t, tl) in TARGETS for (lp, sl) in LPS]
    cols = ["#4C72B0", "#8Fb4d9", "#DD8452", "#f0b98e"]
    fig, ax = plt.subplots(figsize=(1.9*len(feats)+2, 5))
    x = np.arange(len(feats)); w = 0.2
    for k, (t, lp, lab) in enumerate(combos):
        C = all_coefs[(table, t, lp)]
        vals = [C[f]["coef"] for f in feats]
        err = [[C[f]["coef"]-C[f]["lo"] for f in feats], [C[f]["hi"]-C[f]["coef"] for f in feats]]
        bars = ax.bar(x + (k-1.5)*w, vals, w, yerr=err, capsize=3, color=cols[k],
                      edgecolor="black", lw=0.4, label=lab,
                      error_kw={"lw": 1.0, "zorder": 6, "ecolor": "#222222"})
        for f, b, v, e_hi in zip(feats, bars, vals, err[1]):
            bx = b.get_x() + b.get_width()/2
            ax.text(bx, v/2, f"{v:.1f}", ha="center", va="center", rotation=90,
                    fontsize=6.5, fontweight="bold", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.8))
            if C[f]["p"] < 0.05:
                yy = v + e_hi if v >= 0 else v - e_hi
                ax.annotate("*", (bx, yy), ha="center", zorder=8,
                            va="bottom" if v >= 0 else "top", fontsize=11, fontweight="bold")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([FLABEL[f] for f in feats], fontsize=9, rotation=15)
    ax.set_ylabel("OLS coefficient: effect of enabling the component (pp)",
                  fontsize=9)
    ax.set_title(f"Regression coefficients: {table.replace('_', ' ')}  (* p<0.05)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, framealpha=0.8); ax.grid(axis="y", alpha=0.25)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig); print(f"written: {out}.png/.pdf")


def fig_fanova(all_imp, table, feats, out):
    combos = [(t, lp, f"{tl} {sl}") for (t, tl) in TARGETS for (lp, sl) in LPS]
    cols = ["#4C72B0", "#8Fb4d9", "#DD8452", "#f0b98e"]
    fig, ax = plt.subplots(figsize=(1.9*len(feats)+2, 4.6))
    x = np.arange(len(feats)); w = 0.2
    for k, (t, lp, lab) in enumerate(combos):
        I = all_imp[(table, t, lp)]
        vals = [100*I[f] for f in feats]
        bars = ax.bar(x + (k-1.5)*w, vals, w, color=cols[k], edgecolor="black", lw=0.4, label=lab)
        for b, v in zip(bars, vals):
            bx = b.get_x() + b.get_width()/2
            ax.text(bx, v/2, f"{v:.0f}", ha="center", va="center", rotation=90,
                    fontsize=6.5, fontweight="bold", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.8))
    ax.set_xticks(x); ax.set_xticklabels([FLABEL[f] for f in feats], fontsize=9, rotation=15)
    ax.set_ylabel("Relative fANOVA importance (%)", fontsize=9)
    ax.set_title(f"fANOVA (RF permutation importance) — {table.replace('_',' ')}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, framealpha=0.8); ax.grid(axis="y", alpha=0.25)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig); print(f"written: {out}.png/.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(_ROOT, "results", "ablation", "runs.csv"))
    ap.add_argument("--out", default=os.path.join(_ROOT, "results", "component_importance"))
    ap.add_argument("--multiseed", action="store_true",
                    help="use the grid plus the seeds table (cell means); "
                         "writes component_importance_multiseed.csv")
    ap.add_argument("--interactions", action="store_true",
                    help="add the component interactions the design can estimate; "
                         "the plot still shows the isolated effects only")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = load_multiseed(args.csv) if args.multiseed else load(args.csv)
    sfx = ("_multiseed" if args.multiseed else "") + ("_inter" if args.interactions else "")
    if "config_base" not in set(df.config):
        print("  [error] config_base is not in the CSV yet. "
              "Run:  python3 run_ablation.py --configs config_base --include-big")
        return

    all_coefs, all_imp, csv_rows = {}, {}, []
    for table, (cfgs, feats) in TABLES.items():
        for (t, tl) in TARGETS:
            for (lp, sl) in LPS:
                data = build(df, cfgs, lp, t)
                coefs, r2 = ols_coefs(data, feats, interactions=args.interactions)
                if args.interactions and t == TARGETS[0][0] and lp == LPS[0][0]:
                    print(f"  {table}: estimable interactions "
                          f"{interaction_terms(data, feats)}")
                imp = rf_importance(data, feats)
                all_coefs[(table, t, lp)] = coefs
                all_imp[(table, t, lp)] = imp
                for f in feats:
                    csv_rows.append({"table": table, "target": t, "label_pct": lp,
                                     "feature": f, "coef_pp": coefs[f]["coef"],
                                     "ci_lo": coefs[f]["lo"], "ci_hi": coefs[f]["hi"],
                                     "p_value": coefs[f]["p"], "fanova_importance": imp[f],
                                     "ols_r2": r2})
        fig_coef(all_coefs, table, feats,
                 os.path.join(args.out, f"fig_coef_{table}{sfx}"))
        fig_fanova(all_imp, table, feats,
                   os.path.join(args.out, f"fig_fanova_{table}{sfx}"))
    out_csv = os.path.join(args.out, f"component_importance{sfx}.csv")
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
    print(f"written: {out_csv}")


if __name__ == "__main__":
    main()
