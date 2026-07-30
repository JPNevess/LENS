"""Pairwise robustness diagnostics between every pair of methods.

For each pair and each stream, the difference in score; from those, the effect
size, the win rate, the break point, and whether the pair fails any of the three.
Feeds Figures 5b and 5c and the lower rows of the results table.

    python experiments/compute_robustness_metrics.py --multiseed
"""

import argparse
import glob
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import os, argparse, itertools
import numpy as np, pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUDE = ("Rialto", "NOAA", "airlines_without_AirportToFrom", "RBF_a", "ForestCoverType")

TAU_D, TAU_W, TAU_B = 0.2, 0.6, 0.2
TARGETS = [("global_acc", "Accuracy"), ("f1_score", "F1")]
LPS = (5, 1)


def cohens_d(delta):
    sd = float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0
    return float(np.mean(delta)) / sd if sd > 0 else float("inf")


def win_rate(delta):
    return float(np.mean(delta > 0))


def breakdown_point(delta):
    k = len(delta)
    s = float(np.sum(delta))
    if s <= 0:
        return 0.0
    for m, d in enumerate(sorted(delta, reverse=True), start=1):
        s -= d
        if s <= 0:
            return m / k
    return 1.0


def pair_metrics(sub, cfg_a, cfg_b, target):
    a = sub[sub.config == cfg_a].set_index("dataset")[target]
    b = sub[sub.config == cfg_b].set_index("dataset")[target]
    common = a.index.intersection(b.index)
    if len(common) < 3:
        return None
    delta = (a.loc[common] - b.loc[common]).values
    if delta.mean() < 0:
        cfg_a, cfg_b, delta = cfg_b, cfg_a, -delta
    d, wr, bp = cohens_d(delta), win_rate(delta), breakdown_point(delta)
    fragile = (wr <= TAU_W) or (d <= TAU_D) or (bp <= TAU_B)
    return {"winner": cfg_a, "loser": cfg_b, "k_datasets": len(common),
            "mean_gap": float(delta.mean()), "cohens_d": d, "win_rate": wr,
            "breakdown_point": bp,
            "fail_magnitude": d <= TAU_D, "fail_consistency": wr <= TAU_W,
            "fail_stability": bp <= TAU_B, "fragile": fragile}


def load_multiseed(abl_csv, seeds_csv):
    cols = ["dataset", "config", "label_pct", "seed", "global_acc", "f1_score"]
    parts = [pd.read_csv(abl_csv)[cols]]
    if os.path.exists(seeds_csv):
        parts.append(pd.read_csv(seeds_csv)[cols])
    df = pd.concat(parts, ignore_index=True)
    df = df[df.global_acc.notna() & ~df.dataset.isin(EXCLUDE)]
    df = df.drop_duplicates(["dataset", "config", "label_pct", "seed"])
    n_seeds = df.groupby(["dataset", "label_pct"]).seed.nunique()
    print(f"  multi-seed: {df.dataset.nunique()} datasets, "
          f"{n_seeds.min()}-{n_seeds.max()} seeds per cell")
    return df.groupby(["dataset", "config", "label_pct"], as_index=False)[
        ["global_acc", "f1_score"]].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(_ROOT, "results", "ablation", "runs.csv"))
    ap.add_argument("--out", default=os.path.join(_ROOT, "results", "sota_metrics"))
    ap.add_argument("--multiseed", action="store_true",
                    help="average over the seeds in results/seeds/runs.csv "
                         "(escreve sota_metrics_multiseed.csv)")
    ap.add_argument("--seeds-csv", dest="seeds_csv",
                    default=os.path.join(_ROOT, "results", "seeds", "runs.csv"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.multiseed:
        df = load_multiseed(args.csv, args.seeds_csv)
    else:
        df = pd.read_csv(args.csv)
        df = df[df.global_acc.notna() & ~df.dataset.isin(EXCLUDE)]
    configs = sorted(df.config.unique())

    rows = []
    for (t, tl) in TARGETS:
        for lp in LPS:
            sub = df[df.label_pct == lp]
            for cfg_a, cfg_b in itertools.combinations(configs, 2):
                m = pair_metrics(sub, cfg_a, cfg_b, t)
                if m:
                    rows.append({"target": t, "label_pct": lp, **m})
    res = pd.DataFrame(rows)
    name = "sota_metrics_multiseed.csv" if args.multiseed else "sota_metrics.csv"
    res.to_csv(os.path.join(args.out, name), index=False)
    print(f"Guardado: {os.path.join(args.out, name)}  ({len(res)} pares)")

    summ = []
    for (t, tl) in TARGETS:
        for lp in LPS:
            s = res[(res.target == t) & (res.label_pct == lp)]
            if s.empty:
                continue
            vs_sleade = s[(s.winner == "config_0") | (s.loser == "config_0")]
            summ.append({"target": t, "label_pct": lp, "n_pairs": len(s),
                         "fragility_rate": float(s.fragile.mean()),
                         "fragility_vs_sleade": float(vs_sleade.fragile.mean())
                         if len(vs_sleade) else float("nan")})
    summ = pd.DataFrame(summ)
    sname = "sota_summary_multiseed.csv" if args.multiseed else "sota_summary.csv"
    summ.to_csv(os.path.join(args.out, sname), index=False)
    print(f"Guardado: {os.path.join(args.out, sname)}")
    print("\nFragility rate (fraction of pairs failing at least one test):")
    print(summ.to_string(index=False))

    print("\nPares vs SLEADE (config_0) — Accuracy:")
    key = res[(res.target == "global_acc") &
              ((res.winner == "config_0") | (res.loser == "config_0"))]
    cols = ["label_pct", "winner", "loser", "mean_gap", "cohens_d",
            "win_rate", "breakdown_point", "fragile"]
    if not key.empty:
        print(key[cols].sort_values(["label_pct", "mean_gap"], ascending=[False, False])
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
