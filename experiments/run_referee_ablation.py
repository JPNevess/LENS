"""Compare a high-order competence map against a binary relevance one.

The same configurations are run twice, once with a referee that models the
members jointly and once with one independent model per member, so the
difference isolates the effect of modelling dependencies. Feeds Figures 7 and 11.

    python experiments/run_referee_ablation.py
"""

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
PROBE_NAME = "referee"


import numpy as np, pandas as pd

from lens import streams

OUT_DIR      = os.path.join(_ROOT, "results", "referee_ablation")
SAMPLES_DIR  = os.path.join(_ROOT, "results", "probes", PROBE_NAME)
CSV_PATH     = os.path.join(OUT_DIR, "runs.csv")

EXCLUDE = ("Rialto", "NOAA", "airlines_without_AirportToFrom", "RBF_a", "ForestCoverType")

MODES   = ("mlhat", "binary_relevance")
MODE_LABEL = {"mlhat": "MLHAT (multi-label, high-order)",
              "binary_relevance": "Binary Relevance (HAT/membro, first-order)"}
MODE_COL   = {"mlhat": "#4C72B0", "binary_relevance": "#DD8452"}
CONFIGS = ("config_2", "config_12")
PROBE_CONFIG = "config_2"
LPS     = (5, 1)
INK, MUTED = "#222222", "#666666"


def _probe_metrics(est_acc, true_correct):
    estA = est_acc.astype(float)
    tc   = true_correct.astype(float)
    relevance_mae = float(np.mean(np.abs(estA.mean(0) - tc.mean(0))))
    brier         = float(np.mean((estA - tc) ** 2))
    est_c = (estA >= 0.5).astype(float)
    Dt = _disagreement_matrix(tc)
    De = _disagreement_matrix(est_c)
    iu = np.triu_indices(tc.shape[1], k=1)
    dt, de = Dt[iu], De[iu]
    if np.std(dt) > 0 and np.std(de) > 0:
        div_corr = float(np.corrcoef(dt, de)[0, 1])
    else:
        div_corr = np.nan
    div_mae = float(np.mean(np.abs(dt - de)))
    return dict(relevance_mae=relevance_mae, brier=brier,
                div_corr=div_corr, div_mae=div_mae)


def _disagreement_matrix(C):
    N, K = C.shape
    p   = C.mean(0)
    co  = (C.T @ C) / N
    D   = p[:, None] + p[None, :] - 2 * co
    return np.clip(D, 0.0, 1.0)


def _npz(mode, cfg, ds, lp):
    return os.path.join(SAMPLES_DIR, f"{mode}__{cfg}__{ds}__{lp}labels.npz")


def _worker(args):
    mode, cfg, dpath, lp, max_inst, cap = args
    ds = os.path.splitext(os.path.basename(dpath))[0]
    from lens.evaluation import run_experiment
    r = run_experiment(dpath, cfg, label_pct=lp, max_instances=max_inst,
                       verbose=False, referee_mode=mode, referee_probe=True,
                       referee_probe_cap=cap, seed=42)
    p = r["referee_probe"]
    row = {"mode": mode, "config": cfg, "dataset": ds, "label_pct": lp,
           "global_acc": r["global_acc"], "f1_score": r["f1_score"],
           "elapsed_s": r["elapsed_s"]}
    if p is not None:
        np.savez_compressed(_npz(mode, cfg, ds, lp),
                            est_acc=p["est_acc"], true_correct=p["true_correct"],
                            n_total=p["n_total"])
        row.update(_probe_metrics(p["est_acc"], p["true_correct"]))
    return row


def collect(datasets, configs, label_pcts, max_inst, cap, workers):
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    done = set()
    if os.path.exists(CSV_PATH):
        old = pd.read_csv(CSV_PATH)
        done = {(r["mode"], r["config"], r["dataset"], int(r["label_pct"]))
                for _, r in old.iterrows()}
        rows = old.to_dict("records")
    else:
        rows = []
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
        print("  [resume] every ablation run already collected.")
        return pd.DataFrame(rows)
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
                    rows = [r for r in rows if not (
                        r["mode"] == row["mode"] and r["config"] == row["config"]
                        and r["dataset"] == row["dataset"]
                        and int(r["label_pct"]) == row["label_pct"])]
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
                    print(f"  OK [{n}/{len(tasks)}] {row['dataset']:12s} "
                          f"{row['mode']:16s} {row['config']:10s} {row['label_pct']}%  "
                          f"acc={row['global_acc']:.4f}  "
                          f"relMAE={row.get('relevance_mae', float('nan')):.4f}  "
                          f"divR={row.get('div_corr', float('nan')):.3f}  "
                          f"t={time.time()-start:.0f}s")
                except (BrokenProcessPool, Exception) as e:
                    if "Java" in str(e) or isinstance(e, BrokenProcessPool):
                        retry.append(t)
                    else:
                        print(f"  ERR {t[0]} {t[1]} {os.path.basename(t[2])}: {e}")
        pending = retry
    print(f"  recolha terminada em {time.time()-start:.0f}s.")
    return pd.DataFrame(rows)


def _load(mode, cfg, ds, lp):
    p = _npz(mode, cfg, ds, lp)
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z["est_acc"].astype(float), z["true_correct"].astype(float)


def _relevance_scores(mode, datasets, lp):
    from sklearn.metrics import roc_auc_score
    mae_d, auc_d = {}, {}
    for dpath in datasets:
        ds = os.path.splitext(os.path.basename(dpath))[0]
        r = _load(mode, PROBE_CONFIG, ds, lp)
        if r is None:
            continue
        estA, tc = r
        mae_d[ds] = float(np.mean(np.abs(estA.mean(0) - tc.mean(0))))
        s, y = estA.ravel(), tc.ravel().astype(int)
        auc_d[ds] = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else np.nan
    return mae_d, auc_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--label-pcts", type=int, nargs="+", default=list(LPS), dest="label_pcts")
    ap.add_argument("--include-big", action="store_true", dest="include_big")
    ap.add_argument("--max-instances", type=int, default=0, dest="max_instances")
    ap.add_argument("--cap", type=int, default=20000, help="cap on probe rows per run")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--plots-only", action="store_true", dest="plots_only")
    ap.add_argument("--collect-only", action="store_true", dest="collect_only",
                    help="collect only, do not rewrite the summary "
                         "(it would use only this call's configs)")
    args = ap.parse_args()

    names = args.datasets or [n for n in streams.DEFAULT if n not in EXCLUDE]
    paths = streams.resolve_all(
        names, include_big=bool(args.datasets or args.include_big))
    paths = sorted(paths, key=os.path.getsize)
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    configs = tuple(args.configs)
    print(f"Datasets: {names}")
    print(f"Configs: {list(configs)} | modos: {list(MODES)} | labels: {args.label_pcts}")

    if not args.plots_only:
        df = collect(paths, configs, tuple(args.label_pcts),
                     args.max_instances, args.cap, args.workers)
    else:
        df = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame()

    if args.collect_only:
        print("\n  collection finished; the tables are in "
              f"{OUT_DIR}")
        return
    if df.empty:
        print("  [warning] no data collected."); return

    print("\nSummary (mean over streams, 5% labels):")
    g = df[df.label_pct == 5].groupby(["config", "mode"]).agg(
        acc=("global_acc", "mean"), relevance_mae=("relevance_mae", "mean"),
        brier=("brier", "mean"), div_corr=("div_corr", "mean")).round(4)
    print(g.to_string())


if __name__ == "__main__":
    main()
