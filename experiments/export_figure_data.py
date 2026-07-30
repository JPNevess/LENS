"""Reduce the probe archives to the compact tables the figures read.

The studies in this directory write per-instance probe archives, which are large
because they record one row per instance and one column per ensemble member. No
figure needs that resolution: each one plots an aggregate. This script computes
those aggregates once and writes them as CSV, so the figures can be regenerated
without the archives.

    python experiments/export_figure_data.py --source results/probes

Outputs, under ``results/figure_data``:

    diversity_pairs.csv        label-free diversity estimate vs ground truth
    signal_calibration.csv     per-member selection signal vs real accuracy
    window_signals.csv         internal signals per window of the stream
    admission_counts.csv       admitted and correct counts on a threshold grid
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(_ROOT, "results", "figure_data")

DATASETS = ("AGR_a", "AGR_g", "CovtFD", "Electricity", "LED_a", "LED_g",
            "RBF_f", "RBF_m", "airlines")
LABEL_PCTS = (5, 1)

# Streams with known drift positions, used by the window-signal export.
DRIFT_STREAMS = {
    "AGR_a": [20000, 40000, 60000, 80000],
    "LED_a": [20000, 40000, 60000, 80000],
    "AGR_g": [20000, 28000, 43000, 50000, 70000, 72000, 79000],
    "LED_g": [20000, 28000, 43000, 50000, 70000, 72000, 79000],
}
LAMBDA_TAGS = ("w0.50", "w0.75", "w0.95", "adapt")

# Signals whose admission behaviour is compared, and the run that probes each.
ADMISSION_SIGNALS = {
    "config_5": "M",
    "config_6": "c",
    "config_13": "A_hat",
    "config_7": "C",
    "config_8": "w",
}
THRESHOLDS = np.round(np.arange(0.70, 1.0 + 1e-9, 0.0025), 4)


# --------------------------------------------------------------------- helpers

def _correctness_disagreement(correct):
    """Pairwise disagreement on correctness states (eq. 5)."""
    n = correct.shape[0]
    p = correct.mean(0)
    co = (correct.T @ correct) / n
    return np.clip(p[:, None] + p[None, :] - 2 * co, 0.0, 1.0)


def _prediction_disagreement(preds):
    """Pairwise disagreement on raw predicted classes."""
    k = preds.shape[1]
    out = np.zeros((k, k))
    for i in range(k):
        out[i] = (preds != preds[:, [i]]).mean(0)
    return out


def _upper(matrix):
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def _load(path, *keys):
    if not os.path.exists(path):
        return None
    archive = np.load(path, allow_pickle=True)
    return tuple(archive[k] for k in keys)


# ----------------------------------------------------------- figure 1 (top)

def export_diversity_pairs(source, out_dir, per_stream_cap=1500, seed=42):
    """Diversity estimates against ground truth, one row per learner pair.

    Three estimators are compared: plain prediction disagreement, and the
    meta-learned disagreement under a binary-relevance and a high-order
    competence map. Pairs are capped per stream so no single one dominates.
    """
    samples = os.path.join(source, "diversity_comparison", "samples")
    rng = np.random.RandomState(seed)
    rows = []
    for label_pct in LABEL_PCTS:
        for stream in DATASETS:
            for estimator, mode in (("plain", "mlhat"),
                                    ("binary_relevance", "binary_relevance"),
                                    ("high_order", "mlhat")):
                path = os.path.join(
                    samples, f"{mode}__config_2__{stream}__{label_pct}labels.npz")
                loaded = _load(path, "est_acc", "true_correct", "member_preds")
                if loaded is None:
                    continue
                est_acc, true_correct, preds = (loaded[0].astype(float),
                                                loaded[1].astype(float),
                                                loaded[2])
                truth = _upper(_correctness_disagreement(true_correct))
                if estimator == "plain":
                    estimate = _upper(_prediction_disagreement(preds))
                else:
                    estimate = _upper(
                        _correctness_disagreement((est_acc >= 0.5).astype(float)))
                if len(truth) > per_stream_cap:
                    keep = rng.choice(len(truth), per_stream_cap, replace=False)
                    truth, estimate = truth[keep], estimate[keep]
                rows.append(pd.DataFrame({
                    "estimator": estimator, "dataset": stream,
                    "label_pct": label_pct,
                    "ground_truth": truth, "estimate": estimate}))
    return _write(rows, out_dir, "diversity_pairs.csv")


# ---------------------------------------------------------------- figure 7

def export_signal_calibration(source, out_dir, config="config_12"):
    """Per-member selection signal against that member's real accuracy.

    The margin comes from its own probe because the competence probes do not
    keep member identity for it; all three are averaged over the run, so the
    three panels are at the same level of aggregation.
    """
    competence = os.path.join(source, "mlhat_hat_ablation", "samples")
    margins = os.path.join(source, "paper", "mentor17", "samples_margin")
    rows = []
    for label_pct in LABEL_PCTS:
        for stream in DATASETS:
            sources = {
                "margin": (os.path.join(
                    margins, f"{config}__{stream}__{label_pct}labels.npz"),
                    "margins"),
                "binary_relevance": (os.path.join(
                    competence,
                    f"binary_relevance__{config}__{stream}__{label_pct}labels.npz"),
                    "est_acc"),
                "high_order": (os.path.join(
                    competence,
                    f"mlhat__{config}__{stream}__{label_pct}labels.npz"),
                    "est_acc"),
            }
            for signal, (path, key) in sources.items():
                loaded = _load(path, key, "true_correct")
                if loaded is None:
                    continue
                value, correct = loaded[0].astype(float), loaded[1].astype(float)
                rows.append(pd.DataFrame({
                    "signal": signal, "dataset": stream, "label_pct": label_pct,
                    "member": np.arange(value.shape[1]),
                    "signal_mean": value.mean(axis=0),
                    "real_accuracy": correct.mean(axis=0)}))
    return _write(rows, out_dir, "signal_calibration.csv")


# ----------------------------------------------- figures 1 (middle) and 10a

def export_window_signals(source, out_dir, config="config_12", window=1000):
    """Internal signals per window of ``window`` instances.

    Per window: the fraction of members that were correct, pairwise
    disagreement over all pairs, and disagreement restricted to the pairs that
    were both correct. The drift positions are kept so the figures can align
    the windows on them.
    """
    samples = os.path.join(source, "omega_study", "samples")
    rows = []
    for label_pct in LABEL_PCTS:
        for stream, drifts in DRIFT_STREAMS.items():
            for tag in LAMBDA_TAGS:
                path = os.path.join(
                    samples, f"{config}_{stream}_{label_pct}labels_{tag}.npz")
                loaded = _load(path, "true_correct", "member_preds",
                               "instance_idx")
                if loaded is None:
                    continue
                correct = loaded[0].astype(np.int8)
                preds = loaded[1]
                index = loaded[2].astype(np.int64)
                if preds is None or preds.dtype == object:
                    continue
                lo, hi = int(index.min()), int(index.max())
                out = []
                for start in range(lo, hi + 1, window):
                    inside = (index >= start) & (index < start + window)
                    if inside.sum() < 50:
                        continue
                    c_w, p_w = correct[inside], preds[inside]
                    neq = (p_w[:, :, None] != p_w[:, None, :]).mean(axis=0)
                    both = (c_w[:, :, None] & c_w[:, None, :]).astype(float).mean(axis=0)
                    iu = np.triu_indices(neq.shape[0], 1)
                    weights = both[iu]
                    out.append({
                        "center": start + window / 2,
                        "correct_fraction": float(c_w.mean()),
                        "disagreement_all": float(neq[iu].mean()),
                        "disagreement_correct": (
                            float((neq[iu] * weights).sum() / weights.sum())
                            if weights.sum() > 0 else np.nan),
                    })
                if not out:
                    continue
                frame = pd.DataFrame(out)
                frame.insert(0, "lambda_tag", tag)
                frame.insert(0, "label_pct", label_pct)
                frame.insert(0, "dataset", stream)
                frame.insert(0, "config", config)
                frame["drifts"] = ";".join(str(d) for d in drifts)
                rows.append(frame)
    return _write(rows, out_dir, "window_signals.csv")


# ---------------------------------------------------------------- figure 9

def export_admission_counts(source, out_dir):
    """Admitted and correct pseudo-label counts on a grid of thresholds.

    Precision, coverage and risk at any threshold in the grid follow from these
    two counts, which is all the calibration figures need. The high-order and
    binary-relevance variants of the same signal are both exported.
    """
    base = os.path.join(source, "paper", "admission")
    br_competence = os.path.join(source, "mlhat_hat_ablation", "samples")
    rows = []
    for label_pct in LABEL_PCTS:
        for stream in DATASETS:
            candidates = {}
            for config, signal in ADMISSION_SIGNALS.items():
                candidates[(signal, "high_order")] = (
                    os.path.join(base, "samples",
                                 f"{config}__{stream}__{label_pct}labels.npz"),
                    ("score", "correct"))
                candidates[(signal, "binary_relevance")] = (
                    os.path.join(base, "samples_br",
                                 f"{config}__{stream}__{label_pct}labels.npz"),
                    ("score", "correct"))
            candidates[("A_hat", "binary_relevance_probe")] = (
                os.path.join(br_competence,
                             f"binary_relevance__config_13__{stream}__"
                             f"{label_pct}labels.npz"),
                ("est_acc", "true_correct"))

            for (signal, referee), (path, keys) in candidates.items():
                loaded = _load(path, *keys)
                if loaded is None:
                    continue
                score = np.asarray(loaded[0], dtype=float).ravel()
                correct = np.asarray(loaded[1], dtype=int).ravel()
                finite = np.isfinite(score)
                score, correct = score[finite], correct[finite]
                if score.size == 0:
                    continue
                order = np.argsort(-score, kind="mergesort")
                sorted_correct = np.cumsum(correct[order])
                sorted_score = score[order]
                # For each threshold, how many instances score at or above it.
                admitted = np.searchsorted(-sorted_score, -THRESHOLDS,
                                           side="right")
                n_correct = np.where(admitted > 0,
                                     sorted_correct[np.clip(admitted - 1, 0, None)],
                                     0)
                rows.append(pd.DataFrame({
                    "signal": signal, "referee": referee, "dataset": stream,
                    "label_pct": label_pct, "threshold": THRESHOLDS,
                    "n_total": score.size, "n_admitted": admitted,
                    "n_correct": n_correct}))
    return _write(rows, out_dir, "admission_counts.csv")


# ------------------------------------------------------------------------- io

def _write(frames, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    if not frames:
        print(f"  {name:26s} no archives found, skipped")
        return None
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(path, index=False)
    size = os.path.getsize(path) / 1e6
    print(f"  {name:26s} {len(frame):>8d} rows  {size:6.2f} MB")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True,
                        help="directory holding the probe archives")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--only", nargs="+",
                        choices=["diversity", "signals", "windows", "admission"],
                        help="export only these tables")
    args = parser.parse_args()

    wanted = set(args.only) if args.only else {
        "diversity", "signals", "windows", "admission"}
    print(f"source: {args.source}\nout:    {args.out}\n")
    if "diversity" in wanted:
        export_diversity_pairs(args.source, args.out)
    if "signals" in wanted:
        export_signal_calibration(args.source, args.out)
    if "windows" in wanted:
        export_window_signals(args.source, args.out)
    if "admission" in wanted:
        export_admission_counts(args.source, args.out)


if __name__ == "__main__":
    main()
