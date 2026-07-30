"""Validate the signals that decide whether a pseudo-label is admitted.

For every unlabelled instance it records the admission score each signal would
give and whether the pseudo-label it would produce is in fact correct, which
measures the signal itself rather than the whole system. Feeds Figure 9.

    python experiments/run_admission_study.py
"""

import argparse
import glob
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
PROBE_NAME = "admission"


import numpy as np, pandas as pd

DATASETS_DIR = os.path.join(_ROOT, "data")
OUT_DIR      = os.path.join(_ROOT, "results", "admission")
SAMPLES_DIR  = os.path.join(_ROOT, "results", "probes", PROBE_NAME)

EXCLUDE = ("Rialto", "NOAA", "airlines_without_AirportToFrom", "RBF_a", "ForestCoverType")
BIG     = ("CovtFD", "ForestCoverType", "PokerHand")

CONFIG_SIGNAL = {
    "config_5":  ("M", "M = margin  (config 5, self-train)",             "#4C72B0", "o"),
    "config_6":  ("c", "c = learn-by-disagreement  (config 6)",          "#DD8452", "s"),
    "config_13": ("A", "Â = MLHAT accuracy  (config 13, meta-self-train)", "#55A868", "^"),
    "config_7":  ("C", "C = √(Â·M)  (config 7, weighted self-training)",   "#8172B2", "D"),
    "config_8":  ("w", "w = sqrt(A_bar * c), hybrid disagreement", "#C44E52", "v"),
    "config_13_br": ("A(BR)", "A_hat under a binary relevance referee",
                     "#937860", "P"),
    "config_8_br": ("w(BR)", "w under a binary relevance referee",
                    "#DA8BC3", "X"),
    "config_7_br": ("C(BR)", "C under a binary relevance referee",
                    "#64B5CD", "X"),
}
CONFIGS = ("config_5", "config_6", "config_13", "config_7", "config_8")
CONFIGS_BR = ("config_7",)
BR_SUFFIX  = "_br"
ALL_SIGNALS = tuple(CONFIG_SIGNAL.keys())
BR_SAMPLES = os.path.join(_ROOT, "results", "probes", "referee")
SAMPLES_BR = os.path.join(OUT_DIR, "samples_br")
LPS     = (5, 1)
INK, MUTED = "#222222", "#666666"


def _npz_path(cfg, ds, lp, mode="mlhat"):
    d = SAMPLES_DIR if mode == "mlhat" else SAMPLES_BR
    return os.path.join(d, f"{cfg}__{ds}__{lp}labels.npz")


def _worker(args):
    cfg, dpath, lp, max_inst, cap, mode = args
    ds = os.path.splitext(os.path.basename(dpath))[0]
    from lens.evaluation import run_experiment
    r = run_experiment(dpath, cfg, label_pct=lp, max_instances=max_inst,
                       verbose=False, admission=True, admission_cap=cap, seed=42,
                       referee_mode=mode)
    a = r["admission"]
    if a is None:
        return (cfg, ds, lp, 0)
    np.savez_compressed(_npz_path(cfg, ds, lp, mode), score=a["score"],
                        correct=a["correct"], signal=a["signal"],
                        n_total=a["n_total"])
    return (cfg, ds, lp, int(a["score"].size))


def collect(datasets, label_pcts, max_inst, cap, workers, mode="mlhat",
            configs=None):
    out_dir = SAMPLES_DIR if mode == "mlhat" else SAMPLES_BR
    configs = configs or (CONFIGS if mode == "mlhat" else CONFIGS_BR)
    os.makedirs(out_dir, exist_ok=True)
    tasks = []
    for cfg in configs:
        for dpath in datasets:
            ds = os.path.splitext(os.path.basename(dpath))[0]
            for lp in label_pcts:
                if os.path.exists(_npz_path(cfg, ds, lp, mode)):
                    continue
                tasks.append((cfg, dpath, lp, max_inst, cap, mode))
    if not tasks:
        print("  [resume] every admission run already collected.")
        return
    print(f"  {len(tasks)} admission runs to go ({workers} workers)")
    import multiprocessing as _mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool
    start, done = time.time(), 0
    pending = tasks
    for _round in range(3):
        if not pending:
            break
        retry = []
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=_mp.get_context("spawn"),
                                 max_tasks_per_child=1) as ex:
            fut = {ex.submit(_worker, t): t for t in pending}
            for f in as_completed(fut):
                t = fut[f]
                try:
                    cfg, ds, lp, n = f.result()
                    done += 1
                    print(f"  OK [{done}/{len(tasks)}] {ds:14s} {cfg:10s} "
                          f"{lp}%  n={n}  t={time.time()-start:.0f}s")
                except (BrokenProcessPool, Exception) as e:
                    if "Java" in str(e) or isinstance(e, BrokenProcessPool):
                        retry.append(t)
                    else:
                        print(f"  ERR {t[0]} {os.path.basename(t[1])}: {e}")
        pending = retry
    print(f"  recolha terminada em {time.time()-start:.0f}s.")


def _load(cfg, ds, lp):
    if cfg == "config_13_br":
        p = os.path.join(BR_SAMPLES,
                         f"binary_relevance__config_13__{ds}__{lp}labels.npz")
        if not os.path.exists(p):
            return None
        z = np.load(p, allow_pickle=True)
        return (z["est_acc"].astype(float).ravel(),
                z["true_correct"].astype(int).ravel())
    if cfg.endswith(BR_SUFFIX):
        p = _npz_path(cfg[:-len(BR_SUFFIX)], ds, lp, "binary_relevance")
        if not os.path.exists(p):
            return None
        z = np.load(p, allow_pickle=True)
        return z["score"].astype(float), z["correct"].astype(int)
    p = _npz_path(cfg, ds, lp)
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return z["score"].astype(float), z["correct"].astype(int)


def _rc_curve(score, correct):
    o = np.argsort(-score, kind="mergesort")
    s, c = score[o], correct[o]
    n = s.size
    last = np.r_[np.nonzero(np.diff(s))[0], n - 1]
    k = last + 1
    return k / n, 1.0 - np.cumsum(c)[last] / k, s[last]


def _rc_stats(score, correct, tau_lo=0.80, tau_hi=1.00, min_n=50):
    cov, risk, thr = _rc_curve(score, correct)
    grid = np.linspace(1e-4, 1.0, 2001)
    ri = np.interp(grid, cov, risk, left=risk[0], right=risk[-1])
    e = 1.0 - float(correct.mean())
    aurc = float(ri.mean())
    aurc_opt = float(e + (1 - e) * np.log(1 - e)) if 0 < e < 1 else 0.0
    taus = np.linspace(tau_lo, tau_hi, 41)
    rop = []
    for t in taus:
        m = score >= t
        rop.append(1.0 - float(correct[m].mean()) if m.sum() >= min_n else np.nan)
    at = lambda c: float(np.interp(c, cov, risk, left=risk[0], right=risk[-1]))
    return {"aurc": aurc, "aurc_opt": aurc_opt, "eaurc": aurc - aurc_opt,
            "risk_op": float(np.nanmean(rop)) if np.any(np.isfinite(rop)) else np.nan,
            "risk_at_05": at(0.05), "risk_at_10": at(0.10), "risk_at_20": at(0.20),
            "cov_at_09": float((score >= 0.9).mean())}


def _stats(score, correct):
    from scipy.stats import spearmanr, pearsonr
    from sklearn.metrics import roc_auc_score
    out = {"n": int(score.size), "base_rate": float(correct.mean())}
    if len(np.unique(correct)) < 2 or score.std() == 0:
        out.update(spearman=np.nan, pearson=np.nan, roc_auc=np.nan)
        return out
    out["spearman"] = float(spearmanr(score, correct).correlation)
    out["pearson"]  = float(pearsonr(score, correct)[0])
    out["roc_auc"]  = float(roc_auc_score(correct, score))
    out.update(_rc_stats(score, correct))
    return out


def _pool_balanced(cfg, datasets, lp, per_ds_cap=40000, seed=42):
    """Samples from every stream, with the same cap each so that no large stream
    dominates the pooled curves."""
    rng = np.random.RandomState(seed)
    S, C = [], []
    for dpath in datasets:
        ds = os.path.splitext(os.path.basename(dpath))[0]
        r = _load(cfg, ds, lp)
        if r is None:
            continue
        s, c = r
        if s.size > per_ds_cap:
            idx = rng.choice(s.size, per_ds_cap, replace=False)
            s, c = s[idx], c[idx]
        S.append(s); C.append(c)
    if not S:
        return None
    return np.concatenate(S), np.concatenate(C)


def analyze(datasets, label_pcts):
    ds_names = [os.path.splitext(os.path.basename(d))[0] for d in datasets]
    rows = []
    for cfg in ALL_SIGNALS:
        sig = CONFIG_SIGNAL[cfg][0]
        for lp in label_pcts:
            for dpath in datasets:
                ds = os.path.splitext(os.path.basename(dpath))[0]
                r = _load(cfg, ds, lp)
                if r is None:
                    continue
                st = _stats(*r)
                rows.append({"config": cfg, "signal": sig, "dataset": ds,
                             "label_pct": lp, **st})
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "admission_stats.csv"), index=False)
    print(f"written: {os.path.join(OUT_DIR,'stats.csv')}  ({len(df)} rows)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--label-pcts", type=int, nargs="+", default=list(LPS), dest="label_pcts")
    ap.add_argument("--include-big", action="store_true", dest="include_big",
                    help="inclui CovtFD (lento)")
    ap.add_argument("--max-instances", type=int, default=0, dest="max_instances")
    ap.add_argument("--cap", type=int, default=150000, help="cap on samples per run")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--plots-only", action="store_true", dest="plots_only")
    ap.add_argument("--collect-only", action="store_true", dest="collect_only",
                    help="collect only, do not rewrite the summary (use when collecting "
                         "a single label rate) "
                         "com essa coluna)")
    ap.add_argument("--referee-mode", choices=("mlhat", "binary_relevance"),
                    default="mlhat", dest="referee_mode",
                    help="referee used for the collection; binary_relevance writes to "
                         "samples_br/, which feeds the *_br signals of the figures")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="configurations to collect (default: the five MLHAT ones, or "
                         "config_8 under binary_relevance)")
    args = ap.parse_args()

    if args.datasets:
        paths = []
        for n in args.datasets:
            paths += sorted(glob.glob(os.path.join(DATASETS_DIR, f"*{n}*.arff")))
    else:
        paths = sorted(glob.glob(os.path.join(DATASETS_DIR, "*.arff")))
        paths = [p for p in paths
                 if os.path.splitext(os.path.basename(p))[0] not in EXCLUDE]
        if not args.include_big:
            paths = [p for p in paths
                     if os.path.splitext(os.path.basename(p))[0] not in BIG]
    paths = sorted(paths, key=os.path.getsize)
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    print(f"Datasets: {names}")
    _cfgs = args.configs or (CONFIGS if args.referee_mode == "mlhat" else CONFIGS_BR)
    print(f"label rates: {args.label_pcts} | referee: {args.referee_mode} | "
          f"configs: {list(_cfgs)}")

    if not args.plots_only:
        collect(paths, tuple(args.label_pcts), args.max_instances, args.cap,
                args.workers, mode=args.referee_mode, configs=args.configs)

    if args.collect_only:
        print("\n  collection finished; the tables are in "
              f"{OUT_DIR}")
        return

    df = analyze(paths, tuple(args.label_pcts))
    if df.empty:
        print("  [warning] no samples collected.")
        return

    print("\nSummary (mean over streams):")
    cols = [c for c in ["spearman", "pearson", "roc_auc", "base_rate", "aurc",
                        "eaurc", "risk_op", "risk_at_10"] if c in df]
    g = df.groupby(["label_pct", "config", "signal"])[cols].mean().round(3)
    print(g.to_string())

    if "eaurc" in df:
        print("\nE-AURC: paired win rate per stream (lower is better):")
        p = df.pivot_table(index=["label_pct", "dataset"], columns="signal",
                           values="eaurc")
        for a, b in [("C", "M"), ("C", "A"), ("w", "A"), ("w", "c"), ("C", "c")]:
            if a in p and b in p:
                d = (p[a] - p[b]).dropna()
                print(f"   {a:>2s} beats {b:<2s}: {(d < 0).sum():2d}/{len(d)}"
                      f"   mean delta {d.mean():+.3f}")


if __name__ == "__main__":
    main()
