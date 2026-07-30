"""Sweep the relevance/diversity trade-off, and compare it against the dynamic
policy.

Runs the streams with known drift positions at several fixed values of lambda and
with the policy that lowers it after a drift, recording both the rolling accuracy
and the internal signals per window. Feeds Figure 1 (middle and bottom) and
Figures 10a and 10c.

    python experiments/run_lambda_study.py --workers 4
"""

import argparse
import glob
import os
import sys
import time
import warnings

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
PROBE_NAME = "lambda"


warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(_ROOT, "Results", "omega_study")
SAMPLES  = os.path.join(_ROOT, "results", "probes", PROBE_NAME)
HISTORY  = os.path.join(OUT_DIR, "history")
PLOTS    = os.path.join(OUT_DIR, "plots")
CSV_PATH = os.path.join(OUT_DIR, "omega_runs.csv")

import numpy as np
import pandas as pd

OMEGAS      = (0.05, 0.25, 0.50, 0.75, 0.95)
TAGS        = tuple(f"w{o:.2f}" for o in OMEGAS) + ("adapt",)
CONFIGS     = ("config_4", "config_12")
DATASETS    = ("AGR_a", "AGR_g", "LED_a", "LED_g")
LABEL_PCTS  = (5, 1)
SEED        = 42
PROBE_CAP   = 150000
MAX_WORKERS = 4

TRUE_DRIFTS = {
    "AGR_a": [20000, 40000, 60000, 80000],
    "LED_a": [20000, 40000, 60000, 80000],
    "AGR_g": [20000, 28000, 43000, 50000, 70000, 72000, 79000],
    "LED_g": [20000, 28000, 43000, 50000, 70000, 72000, 79000],
}
ABRUPT   = ("AGR_a", "LED_a")
GRADUAL  = ("AGR_g", "LED_g")

CSV_FIELDS = ["dataset", "config", "label_pct", "omega_tag", "omega",
              "adapt_lambda", "global_acc", "f1_score", "drift_count",
              "total_instances", "elapsed_s", "probe_file", "source"]

CFG_LABEL = {"config_4": "selection only",
             "config_12": "(12) Final (MMR + self-training w)"}
INK, MUTED = "#222222", "#666666"


def _tag_omega(tag):
    return float("nan") if tag == "adapt" else float(tag[1:])


def _probe_name(config, ds, lp, tag):
    return f"{config}_{ds}_{lp}labels_{tag}.npz"


def _worker(task):
    """Corre 1 run com ω fixo (ou adapt) + probe; devolve a linha do CSV."""
    ds, config, lp, tag = task["ds"], task["config"], task["lp"], task["tag"]
    print(f"  >> START {ds:8s} {config:10s} {lp:2d}%  ω={tag}  (pid={os.getpid()})",
          flush=True)
    import MMR_DEMS as MD
    res = MD.run_experiment(
        dataset_path       = os.path.join(_ROOT, "datasets", f"{ds}.arff"),
        config             = config,
        label_pct          = lp,
        seed               = SEED,
        verbose            = False,
        max_instances      = task["max_instances"],
        lambda_param       = 0.5 if tag == "adapt" else _tag_omega(tag),
        adapt_lambda       = (tag == "adapt"),
        history_dir        = os.path.join(HISTORY, tag),
        diag_all           = True,
        referee_probe      = True,
        referee_probe_preds= True,
        referee_probe_cap  = PROBE_CAP,
    )
    probe = res.get("referee_probe")
    pname = ""
    if probe is not None:
        pname = _probe_name(config, ds, lp, tag)
        os.makedirs(SAMPLES, exist_ok=True)
        np.savez_compressed(
            os.path.join(SAMPLES, pname),
            true_correct = probe["true_correct"],
            member_preds = probe["member_preds"],
            est_acc      = probe["est_acc"],
            instance_idx = probe["instance_idx"])
    return {"dataset": ds, "config": config, "label_pct": lp,
            "omega_tag": tag, "omega": _tag_omega(tag),
            "adapt_lambda": (tag == "adapt"),
            "global_acc": res["global_acc"], "f1_score": res["f1_score"],
            "drift_count": res["drift_count"],
            "total_instances": res["total_instances"],
            "elapsed_s": round(res["elapsed_s"], 1),
            "probe_file": pname, "source": "omega_study"}


def _append_row(row):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def _done_keys():
    if not os.path.exists(CSV_PATH):
        return set()
    df = pd.read_csv(CSV_PATH)
    return {(r.dataset, r.config, int(r.label_pct), r.omega_tag)
            for r in df.itertuples()}


def seed_from_ablation():
    import shutil
    abl = os.path.join(_ROOT, "Results", "ablation", "ablation_paper.csv")
    if not os.path.exists(abl):
        print("  [seed] grid results not found, nothing to seed from.")
        return
    df = pd.read_csv(abl)
    done = _done_keys()
    n = 0
    for ds in DATASETS:
        for lp in LABEL_PCTS:
            key = (ds, "config_12", lp, "adapt")
            if key in done:
                continue
            hist_src = os.path.join(_ROOT, "Results", ds, "config_12",
                                    f"{lp}labels", f"history_seed{SEED}.csv")
            r = df[(df.dataset == ds) & (df.config == "config_12") &
                   (df.label_pct == lp) & (df.seed == SEED)]
            if not os.path.exists(hist_src) or r.empty:
                continue
            r = r.iloc[-1]
            hist_dst = os.path.join(HISTORY, "adapt", ds, "config_12",
                                    f"{lp}labels")
            os.makedirs(hist_dst, exist_ok=True)
            shutil.copy2(hist_src, os.path.join(hist_dst, f"history_seed{SEED}.csv"))
            _append_row({"dataset": ds, "config": "config_12", "label_pct": lp,
                         "omega_tag": "adapt", "omega": float("nan"),
                         "adapt_lambda": True,
                         "global_acc": r["global_acc"], "f1_score": r["f1_score"],
                         "drift_count": r.get("drift_count", float("nan")),
                         "total_instances": r.get("total_instances", float("nan")),
                         "elapsed_s": float("nan"), "probe_file": "",
                         "source": "run_ablation(seed)"})
            n += 1
    print(f"  [seed] {n} dynamic-lambda cells seeded from the grid.")


def run_all(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    seed_from_ablation()
    done = _done_keys()
    tasks = []
    for config in args.configs:
        for ds in args.datasets:
            for lp in args.label_pcts:
                for tag in TAGS:
                    if (ds, config, lp, tag) in done:
                        continue
                    tasks.append({"ds": ds, "config": config, "lp": lp,
                                  "tag": tag, "max_instances": args.max_instances})
    print(f"\n  {len(tasks)} runs to go | {len(done)} already done (resume) "
          f"| {args.workers} workers\n")
    if not tasks:
        return
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, t): t for t in tasks}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                row = fut.result()
                _append_row(row)
                print(f"  << DONE  {row['dataset']:8s} {row['config']:10s} "
                      f"{row['label_pct']:2d}%  ω={row['omega_tag']}  "
                      f"acc={row['global_acc']:.4f}", flush=True)
            except Exception:
                print(f"  !! FALHOU {t['ds']} {t['config']} {t['lp']}% ω={t['tag']}")
                traceback.print_exc()


def _hist_path(tag, ds, config, lp):
    return os.path.join(HISTORY, tag, ds, config, f"{lp}labels",
                        f"history_seed{SEED}.csv")


def _aligned_curve(vals_by_ds, drifts_by_ds, rel_lo=-5000, rel_hi=15000, step=500):
    grid = np.arange(rel_lo, rel_hi + step, step)
    acc  = np.zeros(len(grid)); cnt = np.zeros(len(grid))
    for ds, (inst, val) in vals_by_ds.items():
        for d in drifts_by_ds[ds]:
            rel = inst - d
            m = (rel >= rel_lo) & (rel <= rel_hi)
            if not m.any():
                continue
            interp = np.interp(grid, rel[m], val[m], left=np.nan, right=np.nan)
            ok = np.isfinite(interp)
            acc[ok] += interp[ok]; cnt[ok] += 1
    with np.errstate(invalid="ignore"):
        return grid, np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


def _probe_window_signals(npz_path, win=1000):
    z = np.load(npz_path, allow_pickle=True)
    C   = z["true_correct"].astype(np.int8)
    P   = z["member_preds"]
    idx = z["instance_idx"].astype(np.int64)
    if P is None or P.dtype == object:
        return None
    lo, hi = int(idx.min()), int(idx.max())
    centers, prop_ok, div_all, div_corr = [], [], [], []
    for start in range(lo, hi + 1, win):
        m = (idx >= start) & (idx < start + win)
        if m.sum() < 50:
            continue
        Cw, Pw = C[m], P[m]
        neq  = (Pw[:, :, None] != Pw[:, None, :]).mean(axis=0)
        both = (Cw[:, :, None] & Cw[:, None, :]).astype(float).mean(axis=0)
        K = neq.shape[0]
        iu = np.triu_indices(K, 1)
        d_all = float(neq[iu].mean())
        w_ij  = both[iu]
        d_cor = float((neq[iu] * w_ij).sum() / w_ij.sum()) if w_ij.sum() > 0 else np.nan
        centers.append(start + win / 2)
        prop_ok.append(float(Cw.mean()))
        div_all.append(d_all)
        div_corr.append(d_cor)
    return (np.array(centers), np.array(prop_ok),
            np.array(div_all), np.array(div_corr))


def _acc_aligned(config, lp, dss, tag):
    """Curva de rolling accuracy (%) alinhada nos drifts, para um ω. None se falta."""
    vals, drifts = {}, {}
    for ds in dss:
        hp = _hist_path(tag, ds, config, lp)
        if not os.path.exists(hp):
            continue
        h = pd.read_csv(hp)
        vals[ds]   = (h["instance"].values.astype(float),
                      h["rolling_acc"].values.astype(float) * 100)
        drifts[ds] = TRUE_DRIFTS[ds]
    if not vals:
        return None, None
    return _aligned_curve(vals, drifts)


def _internal_aligned(config, lp, dss, tag, cache, win=1000):
    """(% membros corretos, diversidade do grupo correto) alinhadas nos drifts."""
    vP, vD, drifts = {}, {}, {}
    for ds in dss:
        sig = cache.get((tag, ds))
        if sig is None:
            continue
        centers, prop_ok, _da, div_corr = sig
        vP[ds] = (centers, prop_ok * 100)
        vD[ds] = (centers, div_corr)
        drifts[ds] = TRUE_DRIFTS[ds]
    if not vP:
        return (None, None), (None, None)
    return _aligned_curve(vP, drifts), _aligned_curve(vD, drifts)


def _smooth(y, k=7):
    y = np.asarray(y, dtype=float)
    out = np.full_like(y, np.nan)
    h = k // 2
    for i in range(len(y)):
        w = y[max(0, i-h):i+h+1]
        w = w[np.isfinite(w)]
        if len(w):
            out[i] = w.mean()
    return out


def _diff_style(ax):
    ax.axhline(0, color="#222222", lw=1.1, zorder=2)
    ax.axvline(0, color="#C44E52", lw=1.4, alpha=0.85, zorder=2)


DCOL = {"0.75-0.50": "#4C72B0", "0.75-0.95": "#DD8452",
        "adapt-0.75": "#55A868", "adapt-0.50": "#8172B2", "adapt-0.95": "#C44E52"}


def make_summary(df):
    rows = []
    for config in CONFIGS:
        for lp in LABEL_PCTS:
            for tag in TAGS:
                sub = df[(df.config == config) & (df.label_pct == lp) &
                         (df.omega_tag == tag)]
                if sub.empty:
                    continue
                pre, early, late = [], [], []
                for ds in DATASETS:
                    hp = _hist_path(tag, ds, config, lp)
                    if not os.path.exists(hp):
                        continue
                    h = pd.read_csv(hp)
                    inst = h["instance"].values
                    acc  = h["rolling_acc"].values
                    for d in TRUE_DRIFTS[ds]:
                        pre.append(acc[(inst >= d - 5000) & (inst < d)].mean())
                        early.append(acc[(inst >= d) & (inst < d + 2000)].mean())
                        late.append(acc[(inst >= d + 2000) & (inst < d + 7500)].mean())
                rows.append({
                    "config": config, "label_pct": lp, "omega_tag": tag,
                    "omega": _tag_omega(tag),
                    "global_acc": sub.groupby("dataset")["global_acc"].mean().mean(),
                    "acc_pre_drift":  np.nanmean(pre)   if pre   else np.nan,
                    "acc_post_0_2k":  np.nanmean(early) if early else np.nan,
                    "acc_post_2k_7k": np.nanmean(late)  if late  else np.nan,
                    "delta_early": (np.nanmean(early) - np.nanmean(pre))
                                   if pre and early else np.nan,
                })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "omega_summary.csv"), index=False)
    print(f"  Guardado: {os.path.join(OUT_DIR, 'omega_summary.csv')}")
    return out


def make_plots():
    if not os.path.exists(CSV_PATH):
        print("  [AVISO] sem omega_runs.csv — corre primeiro as runs.")
        return
    df = pd.read_csv(CSV_PATH)
    df = df[df.global_acc.notna()]
    s = make_summary(df)
    if not s.empty:
        print("\n=== summary (early delta = acc[0,2k) - acc[-5k,0), after vs before) ===")
        print(s.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots-only", action="store_true")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--label-pcts", type=int, nargs="+", default=list(LABEL_PCTS),
                    dest="label_pcts")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--max-instances", type=int, default=0,
                    help="0 = full stream (default)")
    args = ap.parse_args()
    if not args.plots_only:
        run_all(args)
    make_plots()


if __name__ == "__main__":
    main()
