"""Compare label-free diversity estimates against the ground truth.

Runs the ensemble with each referee, records for every instance the estimated
correctness of each member and their raw predictions, and from those computes the
pairwise diversity that each approach would see. Feeds Figure 1 (top).

    python experiments/run_diversity_study.py
"""

import argparse
import glob
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
PROBE_NAME = "diversity"


import numpy as np, pandas as pd

DATASETS_DIR = os.path.join(_ROOT, "data")
OUT_DIR      = os.path.join(_ROOT, "results", "diversity_study")
SAMPLES_DIR  = os.path.join(_ROOT, "results", "probes", PROBE_NAME)
CSV_PATH     = os.path.join(OUT_DIR, "runs.csv")

EXCLUDE = ("Rialto", "NOAA", "airlines_without_AirportToFrom", "RBF_a",
           "ForestCoverType", "PokerHand")

MODES   = ("mlhat", "binary_relevance")
CONFIGS = ("config_2", "config_12")
PROBE_CONFIG = "config_2"
LPS     = (5, 1)
PROBE_CAP = 20000

EST = {
    "plain":       ("Plain prediction disagreement", "#C44E52", "mlhat"),
    "meta_br":     ("Meta-learned (Binary Relevance)", "#DD8452", "binary_relevance"),
    "meta_global": ("Meta-learned (Global / MLHAT)",   "#4C72B0", "mlhat"),
}
INK, MUTED = "#222222", "#666666"


def _corr_disagreement(C):
    N = C.shape[0]
    p  = C.mean(0)
    co = (C.T @ C) / N
    return np.clip(p[:, None] + p[None, :] - 2 * co, 0.0, 1.0)


def _pred_disagreement(P):
    N, K = P.shape
    D = np.zeros((K, K))
    for i in range(K):
        D[i] = (P != P[:, [i]]).mean(0)
    return D


def _triu(D):
    return D[np.triu_indices(D.shape[0], k=1)]


def _npz(mode, cfg, ds, lp):
    return os.path.join(SAMPLES_DIR, f"{mode}__{cfg}__{ds}__{lp}labels.npz")


def _worker(args):
    mode, cfg, dpath, lp, max_inst, cap = args
    ds = os.path.splitext(os.path.basename(dpath))[0]
    from lens.evaluation import run_experiment
    r = run_experiment(dpath, cfg, label_pct=lp, max_instances=max_inst,
                       verbose=False, referee_mode=mode, referee_probe=True,
                       referee_probe_preds=True, referee_probe_cap=cap, seed=42)
    p = r["referee_probe"]
    row = {"mode": mode, "config": cfg, "dataset": ds, "label_pct": lp,
           "global_acc": r["global_acc"], "elapsed_s": r["elapsed_s"]}
    if p is not None and p.get("member_preds") is not None:
        np.savez_compressed(_npz(mode, cfg, ds, lp), est_acc=p["est_acc"],
                            true_correct=p["true_correct"],
                            member_preds=p["member_preds"], n_total=p["n_total"])
        est_c = (p["est_acc"] >= 0.5).astype(float)
        Dgt   = _triu(_corr_disagreement(p["true_correct"].astype(float)))
        Dmeta = _triu(_corr_disagreement(est_c))
        Dpln  = _triu(_pred_disagreement(p["member_preds"]))
        def _c(a, b):
            return float(np.corrcoef(a, b)[0, 1]) if (a.std() > 0 and b.std() > 0) else np.nan
        row.update(corr_meta_gt=_c(Dmeta, Dgt), mae_meta_gt=float(np.mean(np.abs(Dmeta-Dgt))),
                   corr_plain_gt=_c(Dpln, Dgt), mae_plain_gt=float(np.mean(np.abs(Dpln-Dgt))))
    return row


def collect(datasets, configs, label_pcts, max_inst, cap, workers):
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    done, rows = set(), []
    if os.path.exists(CSV_PATH):
        old = pd.read_csv(CSV_PATH)
        rows = old.to_dict("records")
        done = {(r["mode"], r["config"], r["dataset"], int(r["label_pct"])) for r in rows}
    tasks = []
    for mode in MODES:
        for cfg in configs:
            for dpath in datasets:
                ds = os.path.splitext(os.path.basename(dpath))[0]
                for lp in label_pcts:
                    if (mode, cfg, ds, lp) in done and os.path.exists(_npz(mode, cfg, ds, lp)):
                        continue
                    tasks.append((mode, cfg, dpath, lp, max_inst, cap))
    if not tasks:
        print("  [resume] every run already collected."); return pd.DataFrame(rows)
    print(f"  {len(tasks)} runs a executar ({workers} workers)...")
    import multiprocessing as _mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool
    start, n = time.time(), 0
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
                    row = f.result(); n += 1
                    rows = [r for r in rows if not (r["mode"] == row["mode"]
                            and r["config"] == row["config"] and r["dataset"] == row["dataset"]
                            and int(r["label_pct"]) == row["label_pct"])]
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
                    print(f"  OK [{n}/{len(tasks)}] {row['dataset']:12s} {row['mode']:16s} "
                          f"{row['config']:10s} {row['label_pct']}%  "
                          f"corr_meta={row.get('corr_meta_gt', float('nan')):.3f} "
                          f"corr_plain={row.get('corr_plain_gt', float('nan')):.3f}  "
                          f"t={time.time()-start:.0f}s")
                except (BrokenProcessPool, Exception) as e:
                    if "Java" in str(e) or isinstance(e, BrokenProcessPool):
                        retry.append(t)
                    else:
                        print(f"  ERR {t[0]} {os.path.basename(t[2])}: {e}")
        pending = retry
    print(f"  recolha terminada em {time.time()-start:.0f}s.")
    return pd.DataFrame(rows)


def _load(mode, cfg, ds, lp):
    p = _npz(mode, cfg, ds, lp)
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z["est_acc"].astype(float), z["true_correct"].astype(float), z["member_preds"]


def _pairs(est_key, datasets, lp, per_ds_cap=1500, seed=42):
    """(estimate, ground truth) pairs stacked over streams, same cap for each."""
    rng = np.random.RandomState(seed)
    mode = EST[est_key][2]
    X, Y = [], []
    for dpath in datasets:
        ds = os.path.splitext(os.path.basename(dpath))[0]
        r = _load(mode, PROBE_CONFIG, ds, lp)
        if r is None:
            continue
        estA, tc, preds = r
        gt = _triu(_corr_disagreement(tc))
        if est_key == "plain":
            est = _triu(_pred_disagreement(preds))
        else:
            est = _triu(_corr_disagreement((estA >= 0.5).astype(float)))
        if len(gt) > per_ds_cap:
            idx = rng.choice(len(gt), per_ds_cap, replace=False)
            gt, est = gt[idx], est[idx]
        X.append(est); Y.append(gt)
    if not X:
        return None
    return np.concatenate(X), np.concatenate(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--label-pcts", type=int, nargs="+", default=list(LPS), dest="label_pcts")
    ap.add_argument("--max-instances", type=int, default=0, dest="max_instances",
                    help="0 = stream COMPLETO (default); >0 = capar")
    ap.add_argument("--cap", type=int, default=PROBE_CAP, help="cap on probe rows per run")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--plots-only", action="store_true", dest="plots_only")
    args = ap.parse_args()

    if args.datasets:
        paths = []
        for nm in args.datasets:
            paths += sorted(glob.glob(os.path.join(DATASETS_DIR, f"*{nm}*.arff")))
    else:
        paths = [p for p in sorted(glob.glob(os.path.join(DATASETS_DIR, "*.arff")))
                 if os.path.splitext(os.path.basename(p))[0] not in EXCLUDE]
    paths = sorted(paths, key=os.path.getsize)
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    configs = tuple(args.configs)
    print(f"Datasets: {names}")
    print(f"Configs: {list(configs)} | modos: {list(MODES)} | labels: {args.label_pcts} "
          f"| max_inst: {args.max_instances or 'full'}")

    if not args.plots_only:
        df = collect(paths, configs, tuple(args.label_pcts),
                     args.max_instances, args.cap, args.workers)
    else:
        df = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame()
    if df.empty:
        print("  [warning] no data collected."); return

    print("\nFidelity of the estimated diversity against the truth "
          "(mean over streams):")
    s = df[df.config == PROBE_CONFIG]
    for lp in sorted(df.label_pct.unique()):
        ml = s[(s["mode"] == "mlhat") & (s.label_pct == lp)]
        br = s[(s["mode"] == "binary_relevance") & (s.label_pct == lp)]
        print(f"  {lp}% labels  |  correlation:  plain={ml['corr_plain_gt'].mean():.3f}  "
              f"meta-BR={br['corr_meta_gt'].mean():.3f}  meta-Global={ml['corr_meta_gt'].mean():.3f}")
        print(f"             |  MAE↓:        plain={ml['mae_plain_gt'].mean():.3f}  "
              f"meta-BR={br['mae_meta_gt'].mean():.3f}  meta-Global={ml['mae_meta_gt'].mean():.3f}")


if __name__ == "__main__":
    main()
