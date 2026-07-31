"""Compare drift detection signals under label scarcity.

Five strategies, each under a high-order and a binary relevance referee, at both
label rates, all monitored with ADWIN. An alarm counts as a detection if it falls
in the window after a true drift and as a false positive otherwise. The ADWIN
delta is swept and tuned per detector. Feeds Figure 10d.

    python experiments/run_drift_detection.py
"""

import argparse
import glob
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
PROBE_NAME = "drift_detection"


import os, sys, csv, time, argparse, warnings, traceback
warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(_ROOT, "results", "drift_detection")
ALARMS   = os.path.join(OUT_DIR, "alarms")
PLOTS    = os.path.join(OUT_DIR, "plots")
CSV_PATH = os.path.join(OUT_DIR, "runs.csv")

import numpy as np
import pandas as pd

K_ENSEMBLE    = 70
SUBSPACE_FRAC = 0.6
POISSON       = 6.0
GRACE, TIE, CONF = 50, 0.05, 0.01
ADWIN_DELTA   = 0.002
DELTAS        = (0.0001, 0.001, 0.002, 0.01, 0.05, 0.1, 0.3)
SEEDS         = (101,)
DATASETS      = ("AGR_a", "AGR_g", "LED_a", "LED_g")
FA_IGNORE     = 5000
W_ABRUPT, W_GRADUAL = 7500, 9000
GRAD_SHIFT    = 500

TRUE_DRIFTS = {
    "AGR_a": [20000, 40000, 60000, 80000],
    "LED_a": [20000, 40000, 60000, 80000],
    "AGR_g": [20000, 28000, 43000, 50000, 70000, 72000, 79000],
    "LED_g": [20000, 28000, 43000, 50000, 70000, 72000, 79000],
}
GRADUAL = ("AGR_g", "LED_g")

BASE_DETS = [
    ("sup",             "Supervised",                 "supervised"),
    ("plain_dis",       "Plain-Disagreement",         "plain-disagreement"),
    ("studd_mlhat",     "STUDD — MLHAT",              "STUDD"),
    ("studd_br",        "STUDD — BR-HAT",             "STUDD"),
    ("meta_acc_mlhat",  "Meta-Supervised — MLHAT",    "meta-supervised"),
    ("meta_acc_br",     "Meta-Supervised — BR-HAT",   "meta-supervised"),
    ("meta_dis_mlhat",  "Meta-Disagreement — MLHAT",  "meta-disagreement"),
    ("meta_dis_br",     "Meta-Disagreement — BR-HAT", "meta-disagreement"),
]
EXTRA_DETS = [("studd_hat", "STUDD — single HAT (classic)", "STUDD")]
BASE_KEYS  = [b for b, _, _ in BASE_DETS]
SAVE_KEYS  = BASE_KEYS + [b for b, _, _ in EXTRA_DETS]

DETECTORS = [(f"{b}_{lp}", f"{lab} ({lp}%)", fam)
             for lp in (5, 1) for b, lab, fam in BASE_DETS]
DET_KEYS = [d[0] for d in DETECTORS]
DET_LABEL = {k: l for k, l, _ in DETECTORS}
FAM_COLOR = {"supervised": "#4C72B0", "plain-disagreement": "#55A868",
             "STUDD": "#937860", "meta-supervised": "#DD8452",
             "meta-disagreement": "#8172B2"}
DET_FAM = {k: f for k, _, f in DETECTORS}
INK, MUTED = "#222222", "#666666"

CSV_FIELDS = ["dataset", "seed", "label_pct", "n_labeled", "n_instances",
              "elapsed_s", "deltas"] + \
             [f"n_alarms_{k}" for k in SAVE_KEYS]
LABEL_PCTS = (5, 1)
EVERY = {5: 20, 1: 100}


def _npz_path(ds, seed, lp):
    return os.path.join(ALARMS, f"{ds}_seed{int(seed)}_{int(lp)}labels.npz")


def _dkey(delta):
    return "d" + f"{delta:g}".replace(".", "p").replace("-", "m")


class _Adwin:
    def __init__(self, ADWIN, delta):
        self.d = ADWIN(delta=delta)
        for m in ("add_element", "update", "add_input"):
            if hasattr(self.d, m):
                self._upd = getattr(self.d, m)
                break
        else:
            raise RuntimeError("ADWIN has no update method")
        dc = getattr(self.d, "detected_change", None)
        if callable(dc):
            self._det = dc
        elif dc is not None:
            self._det = lambda: self.d.detected_change
        else:
            self._det = lambda: self.d.change_detected

    def add(self, v) -> bool:
        self._upd(float(v))
        return bool(self._det())


def _expected_pairwise_dis(A):
    K = A.size
    s = A.sum() * (1.0 - A).sum() - (A * (1.0 - A)).sum()
    return float(s * 2.0 / (K * (K - 1)))


def _run_dataset(task):
    ds, seed, max_inst = task["ds"], task["seed"], task["max_instances"]
    lp = int(task["label_pct"]); every = EVERY[lp]
    print(f"  >> START {ds:8s} seed={seed} {lp}% labels  (pid={os.getpid()})",
          flush=True)

    from lens import streams
    from lens.ensemble import (ARFFStream, HoeffdingAdaptiveTree, ADWIN,
                          LabeledInstance, MLHAT, _BinaryRelevance,
                          _river_tree, _train_weighted)

    rng    = np.random.RandomState(seed)
    np.random.seed(seed)
    import random as _random; _random.seed(seed)

    stream = ARFFStream(streams.resolve(ds))
    schema = stream.get_schema()
    nfeat  = schema.get_num_attributes()
    ssize  = max(1, int(round(SUBSPACE_FRAC * nfeat)))

    members = [HoeffdingAdaptiveTree(schema=schema, grace_period=GRACE,
                                     tie_threshold=TIE, confidence=CONF)
               for _ in range(K_ENSEMBLE)]
    masks = []
    for _ in range(K_ENSEMBLE):
        m = np.zeros(nfeat); m[rng.choice(nfeat, ssize, replace=False)] = 1.0
        masks.append(m)

    student = HoeffdingAdaptiveTree(schema=schema, grace_period=GRACE,
                                    tie_threshold=TIE, confidence=CONF)
    student_ready = False

    def _mk_br():
        return _BinaryRelevance(_river_tree.HoeffdingAdaptiveTreeClassifier(
            grace_period=200, seed=seed))
    referees = {"mlhat": MLHAT(), "br": _mk_br()}
    studd = {"mlhat": MLHAT(), "br": _mk_br()}

    adwins = {(k, d): _Adwin(ADWIN, d) for k in SAVE_KEYS for d in DELTAS}
    alarms = {(k, d): [] for k in SAVE_KEYS for d in DELTAS}
    sig_v = {k: [] for k in SAVE_KEYS}
    sig_i = {k: [] for k in SAVE_KEYS}

    def _p1(model, x_dict):
        try:
            proba = model.predict_proba_one(x_dict)
        except Exception:
            proba = {}
        P = np.full(K_ENSEMBLE, 0.5)
        for i in range(K_ENSEMBLE):
            p = proba.get(f"model_{i}")
            if p:
                P[i] = p.get(1, 0.0)
        return P

    t, t0, n_lab = 0, time.time(), 0
    while stream.has_more_instances():
        inst = stream.next_instance()
        y    = int(inst.y_index)
        x    = inst.x
        x_dict = {f"feature_{i}": v for i, v in enumerate(x)}

        preds = np.zeros(K_ENSEMBLE, dtype=int)
        for i, m in enumerate(members):
            try:
                minst = LabeledInstance.from_array(schema, x * masks[i], 0)
                votes = np.array(list(
                    m.moa_learner.getVotesForInstance(minst.java_instance)),
                    dtype=float)
                preds[i] = int(votes.argmax()) if votes.size and votes.sum() > 0 else 0
            except Exception:
                preds[i] = 0
        counts = np.bincount(preds)
        vote   = int(counts.argmax())
        plain_dis = 1.0 - float((counts * (counts - 1)).sum()) \
            / (K_ENSEMBLE * (K_ENSEMBLE - 1))

        if student_ready:
            try:
                sv = np.array(list(student.moa_learner.getVotesForInstance(
                    inst.java_instance)), dtype=float)
                st_pred = int(sv.argmax()) if sv.size and sv.sum() > 0 else 0
            except Exception:
                st_pred = 0
        else:
            st_pred = -1

        labeled = (t % every == 0)

        A = {tag: 1.0 - _p1(referees[tag], x_dict) for tag in ("mlhat", "br")}
        D = {tag: _p1(studd[tag], x_dict)          for tag in ("mlhat", "br")}

        feeds = [
            ("sup",            float(vote == y),                       labeled),
            ("plain_dis",      plain_dis,                              True),
            ("studd_mlhat",    float(D["mlhat"].mean()),               True),
            ("studd_br",       float(D["br"].mean()),                  True),
            ("meta_acc_mlhat", float(A["mlhat"].mean()),               True),
            ("meta_acc_br",    float(A["br"].mean()),                  True),
            ("meta_dis_mlhat", _expected_pairwise_dis(A["mlhat"]),     True),
            ("meta_dis_br",    _expected_pairwise_dis(A["br"]),        True),
            ("studd_hat",      float(st_pred == vote),                 st_pred >= 0),
        ]
        for key, val, feed in feeds:
            if not feed:
                continue
            sig_v[key].append(val); sig_i[key].append(t)
            for d in DELTAS:
                if adwins[(key, d)].add(val):
                    alarms[(key, d)].append(t)

        dis_vote = {f"model_{i}": int(preds[i] != vote) for i in range(K_ENSEMBLE)}
        for tag, mdl in studd.items():
            try:
                mdl.learn_one(x_dict, dis_vote)
            except Exception:
                pass
        try:
            student.train(LabeledInstance.from_array(schema, x, vote))
            student_ready = True
        except Exception:
            pass
        if labeled:
            err = {f"model_{i}": int(preds[i] != y) for i in range(K_ENSEMBLE)}
            for tag, ref in referees.items():
                try:
                    ref.learn_one(x_dict, err)
                except Exception:
                    pass
            for i, m in enumerate(members):
                w = int(rng.poisson(POISSON))
                if w > 0:
                    _train_weighted(m, LabeledInstance.from_array(
                        schema, x * masks[i], y), w)
            n_lab += 1

        t += 1
        if t % 5000 == 0:
            print(f"     [{ds} seed={seed} {lp}%] {t} inst  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if max_inst > 0 and t >= max_inst:
            break

    os.makedirs(ALARMS, exist_ok=True)
    out = {"n_instances": t, "n_labeled": n_lab, "label_pct": lp,
           "deltas": np.asarray(DELTAS, dtype=float)}
    for k in SAVE_KEYS:
        out[f"sigv_{k}"] = np.asarray(sig_v[k], dtype=np.float32)
        out[f"sigi_{k}"] = np.asarray(sig_i[k], dtype=np.int64)
        for d in DELTAS:
            out[f"alarms_{k}__{_dkey(d)}"] = np.asarray(alarms[(k, d)], dtype=np.int64)
    np.savez_compressed(_npz_path(ds, seed, lp), **out)
    row = {"dataset": ds, "seed": seed, "label_pct": lp, "n_labeled": n_lab,
           "n_instances": t, "elapsed_s": round(time.time() - t0, 1),
           "deltas": " ".join(f"{d:g}" for d in DELTAS)}
    row.update({f"n_alarms_{k}": len(alarms[(k, ADWIN_DELTA)]) for k in SAVE_KEYS})
    return row


def _has_all_deltas(ds, seed, lp):
    """Is the archive for this (stream, seed, label rate) complete, that is does it
    hold both the alarms for every delta and the signals?
    """
    p = _npz_path(ds, seed, lp)
    if not os.path.exists(p):
        return False
    with np.load(p) as z:
        need = [f"alarms_{k}__{_dkey(d)}" for k in BASE_KEYS for d in DELTAS]
        need += [f"sigv_{k}" for k in BASE_KEYS]
        return all(n in z for n in need)


def _done_keys():
    if not os.path.exists(CSV_PATH):
        return set()
    df = pd.read_csv(CSV_PATH)
    if "label_pct" not in df.columns:
        return set()
    return {(r.dataset, int(r.seed), int(r.label_pct)) for r in df.itertuples()
            if _has_all_deltas(r.dataset, r.seed, r.label_pct)}


def run_all(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    done = _done_keys()
    tasks = [{"ds": ds, "seed": s, "label_pct": lp,
              "max_instances": args.max_instances}
             for ds in args.datasets for s in args.seeds
             for lp in args.label_pcts if (ds, s, lp) not in done]
    print(f"\n  {len(tasks)} passes a executar | {len(done)} feitos (resume) "
          f"| {args.workers} workers")
    print(f"  detetores por passe: {len(BASE_KEYS)} (+1 extra) | "
          f"total reportado: {len(DET_KEYS)}\n")
    if not tasks:
        return
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_dataset, tk): tk for tk in tasks}
        for fut in as_completed(futs):
            tk = futs[fut]
            try:
                row = fut.result()
                new = not os.path.exists(CSV_PATH)
                with open(CSV_PATH, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                    if new:
                        w.writeheader()
                    w.writerow(row)
                print(f"  << DONE {row['dataset']} seed={row['seed']} "
                      f"{row['label_pct']}%  ({row['elapsed_s']}s)", flush=True)
            except Exception:
                print(f"  !! FALHOU {tk['ds']} seed={tk['seed']} "
                      f"{tk['label_pct']}%")
                traceback.print_exc()


def _windows(ds, n):
    drifts = TRUE_DRIFTS[ds]
    W      = W_GRADUAL if ds in GRADUAL else W_ABRUPT
    shift  = GRAD_SHIFT if ds in GRADUAL else 0
    wins = []
    for k, d in enumerate(drifts):
        start = d - shift
        end   = min(start + W, n)
        if k + 1 < len(drifts):
            end = min(end, drifts[k + 1] - shift)
        wins.append((start, end))
    return wins


def _evaluate(alarms, ds, n):
    al = np.asarray([a for a in alarms if a >= FA_IGNORE], dtype=np.int64)
    wins = _windows(ds, n)
    detected, delays = 0, []
    in_any = np.zeros(len(al), dtype=bool)
    for (start, end) in wins:
        m = (al >= start) & (al < end)
        in_any |= m
        if m.any():
            detected += 1
            delays.append(int(al[m].min()) - start)
    fa = int((~in_any).sum())
    da = detected / len(wins) if wins else np.nan
    mtd = float(np.mean(delays)) if delays else np.nan
    return da, mtd, fa, detected, len(wins) - detected


def _prf1(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    rec  = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1   = (2 * prec * rec / (prec + rec)
            if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0 else 0.0)
    return prec, rec, f1


def _score(g, ds):
    worst = W_GRADUAL if ds in GRADUAL else W_ABRUPT
    mtd = g["MTD"].fillna(worst)
    mtd_n = (mtd - mtd.min()) / (mtd.max() - mtd.min()) \
        if mtd.max() > mtd.min() else 0.0
    fa_n = (g["FA"] - g["FA"].min()) / (g["FA"].max() - g["FA"].min()) \
        if g["FA"].max() > g["FA"].min() else 0.0
    return 0.5 * g["DA"] + 0.3 * (1 - fa_n) + 0.2 * (1 - mtd_n)


TUNE_BY = "F1"
def compute_metrics(tune_by=None):
    tune_by = tune_by or TUNE_BY
    if not os.path.exists(CSV_PATH):
        print("  [warning] no runs.csv; run the sweep first.")
        return pd.DataFrame(), pd.DataFrame()
    runs = pd.read_csv(CSV_PATH)
    if "label_pct" not in runs.columns:
        print("  [note] runs table is from an older design "
              "(no label_pct column); re-run the sweep.")
        return pd.DataFrame(), pd.DataFrame()
    runs = runs.drop_duplicates(["dataset", "seed", "label_pct"], keep="last")
    rows = []
    for r in runs.itertuples():
        lp  = int(r.label_pct)
        npz = _npz_path(r.dataset, r.seed, lp)
        if not os.path.exists(npz):
            continue
        with np.load(npz) as z:
            n = int(z["n_instances"])
            deltas = [float(d) for d in z["deltas"]] if "deltas" in z else [ADWIN_DELTA]
            for base in BASE_KEYS:
                det = f"{base}_{lp}"
                for d in deltas:
                    key = f"alarms_{base}__{_dkey(d)}"
                    if key not in z:
                        continue
                    da, mtd, fa, tp, fn = _evaluate(z[key].tolist(), r.dataset, n)
                    prec, rec, f1 = _prf1(tp, fa, fn)
                    rows.append({"dataset": r.dataset, "seed": int(r.seed),
                                 "label_pct": lp, "detector": det, "delta": d,
                                 "DA": da, "MTD": mtd, "FA": fa,
                                 "TP": tp, "FN": fn,
                                 "precision": prec, "recall": rec, "F1": f1})
    allm = pd.DataFrame(rows)
    if allm.empty:
        return allm, allm

    parts = []
    for (ds, seed), g in allm.groupby(["dataset", "seed"]):
        g = g.copy(); g["score_sel"] = _score(g, ds); parts.append(g)
    allm = pd.concat(parts, ignore_index=True)
    allm.to_csv(os.path.join(OUT_DIR, "delta_sweep.csv"), index=False)
    crit = "score_sel" if tune_by == "score" else "F1"
    best = (allm.groupby(["detector", "delta"])[crit].mean()
                .reset_index().sort_values(crit, ascending=False)
                .groupby("detector").first().reset_index()
                .rename(columns={"delta": "best_delta"})[["detector", "best_delta"]])
    print(f"  delta afinado por: {crit}")

    df = allm.merge(best, on="detector")
    df = df[np.isclose(df.delta, df.best_delta)].drop(columns=["score_sel"])
    parts = []
    for (ds, seed), g in df.groupby(["dataset", "seed"]):
        g = g.copy(); g["score"] = _score(g, ds); parts.append(g)
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False)

    summ = (df.assign(tipo=df.dataset.map(
                lambda d: "gradual" if d in GRADUAL else "abrupto"))
              .groupby(["detector", "tipo"])
              .agg(delta=("best_delta", "first"), DA=("DA", "mean"),
                   MTD=("MTD", "mean"), FA=("FA", "mean"),
                   precision=("precision", "mean"), F1=("F1", "mean"),
                   score=("score", "mean"))
              .reset_index())
    summ.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    print("  written: delta_sweep.csv, metrics.csv, summary.csv")
    return df, allm


def _mean_ci(v, conf=0.95):
    from scipy import stats
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan
    if v.size == 1:
        return float(v[0]), 0.0
    half = stats.t.ppf(0.5 + conf/2, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size)
    return float(v.mean()), float(half)


def _split_key(k):
    """'meta_dis_mlhat_5' -> ('meta_dis_mlhat', 5)  (o npz guarda o nome base)."""
    base, lp = k.rsplit("_", 1)
    return base, int(lp)


def _load_lps(ds, seed):
    """{label_pct: npz}, one archive per label rate."""
    out = {}
    for lp in LABEL_PCTS:
        p = _npz_path(ds, seed, lp)
        if os.path.exists(p):
            out[lp] = np.load(p)
    return out


def make_plots():
    df, allm = compute_metrics()
    if df.empty:
        return
    best_delta = dict(zip(df.detector, df.best_delta))
    print("\n=== delta chosen per detector ===")
    for k in DET_KEYS:
        if k in best_delta:
            print(f"  {DET_LABEL[k]:42s} δ = {best_delta[k]:g}")
    print("\n=== summary per detector (mean over abrupt | gradual) ===")
    print(pd.read_csv(os.path.join(OUT_DIR, "summary.csv"))
          .to_string(index=False))


def replay_deltas(new_deltas):
    from lens.ensemble import ADWIN
    for f in sorted(os.listdir(ALARMS)):
        if not f.endswith(".npz"):
            continue
        p = os.path.join(ALARMS, f)
        with np.load(p) as z:
            data = {k: z[k] for k in z.files}
        if f"sigv_{SAVE_KEYS[0]}" not in data:
            print(f"  [skip] {f}: no stored signals (old format)")
            continue
        have = set(float(d) for d in data.get("deltas", []))
        todo = [d for d in new_deltas if d not in have]
        if not todo:
            print(f"  [ok]   {f}: nothing to add")
            continue
        for k in SAVE_KEYS:
            vals, idxs = data[f"sigv_{k}"], data[f"sigi_{k}"]
            for d in todo:
                a = _Adwin(ADWIN, d)
                data[f"alarms_{k}__{_dkey(d)}"] = np.asarray(
                    [int(i) for v, i in zip(vals, idxs) if a.add(float(v))],
                    dtype=np.int64)
        data["deltas"] = np.asarray(sorted(have | set(todo)), dtype=float)
        np.savez_compressed(p, **data)
        print(f"  [+]    {f}: +{len(todo)} deltas → {len(data['deltas'])} no total")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots-only", action="store_true")
    ap.add_argument("--replay-deltas", type=float, nargs="+", default=None,
                    dest="replay_deltas",
                    help="add more deltas from the stored signals, without "
                         "passing over the stream again")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--label-pcts", type=int, nargs="+", default=list(LABEL_PCTS),
                    dest="label_pcts", help="label rates to evaluate")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-instances", type=int, default=0,
                    help="0 = full stream (default)")
    ap.add_argument("--tune-by", choices=("F1", "score"), default=TUNE_BY,
                    dest="tune_by",
                    help="criterion used to pick the ADWIN delta per detector "
                         "(F1 is what the figures report)")
    args = ap.parse_args()
    globals()["TUNE_BY"] = args.tune_by
    if args.replay_deltas:
        replay_deltas(args.replay_deltas)
    elif not args.plots_only:
        run_all(args)
    make_plots()


if __name__ == "__main__":
    main()
