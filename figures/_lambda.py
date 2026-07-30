"""Drift-aligned curves for the lambda study, shared by Figures 1 and 10.

Each stream has several drifts at known positions. A curve is cut around every
drift, resampled onto a common grid of offsets, and averaged over drifts and
streams, so a single curve shows the average behaviour before and after a change
rather than one arbitrary occurrence.

Reads: results/figure_data/window_signals.csv and results/lambda_study/history
"""
import os

import numpy as np
import pandas as pd

from _common import FIGURE_DATA, LAMBDA_HISTORY

WINDOW_CSV = os.path.join(FIGURE_DATA, "window_signals.csv")

ABRUPT = ("AGR_a", "LED_a")
GRADUAL = ("AGR_g", "LED_g")
SEED = 42

# Fixed values of the relevance/diversity trade-off, plus the dynamic policy.
LAMBDA_COLOUR = {"w0.50": "#4C72B0", "w0.75": "#55A868", "w0.95": "#C44E52"}
LAMBDA_NAME = {"w0.50": "Diversity (λ = 0.5)", "w0.75": "Balanced (λ = 0.75)",
               "w0.95": "Relevance (λ = 0.95)", "adapt": "Dynamic λ"}
SHORT_NAME = {"w0.50": "Diversity", "w0.75": "Balanced", "w0.95": "Relevance",
              "adapt": "Dynamic"}


def align(values_by_stream, drifts_by_stream, lo=-5000, hi=15000, step=500):
    """Average a set of series after aligning them on the drift positions."""
    grid = np.arange(lo, hi + step, step)
    total = np.zeros(len(grid))
    count = np.zeros(len(grid))
    for stream, (positions, values) in values_by_stream.items():
        for drift in drifts_by_stream[stream]:
            offset = positions - drift
            inside = (offset >= lo) & (offset <= hi)
            if not inside.any():
                continue
            resampled = np.interp(grid, offset[inside], values[inside],
                                  left=np.nan, right=np.nan)
            ok = np.isfinite(resampled)
            total[ok] += resampled[ok]
            count[ok] += 1
    with np.errstate(invalid="ignore"):
        return grid, np.where(count > 0, total / np.maximum(count, 1), np.nan)


def load_windows():
    return pd.read_csv(WINDOW_CSV)


def internal_aligned(windows, config, label_pct, streams, tag):
    """Aligned (correct members %, disagreement among correct pairs)."""
    correct, diverse, drifts = {}, {}, {}
    sub = windows[(windows.config == config) & (windows.label_pct == label_pct)
                  & (windows.lambda_tag == tag)]
    for stream in streams:
        rows = sub[sub.dataset == stream]
        if rows.empty:
            continue
        centers = rows.center.values.astype(float)
        correct[stream] = (centers, rows.correct_fraction.values * 100)
        diverse[stream] = (centers, rows.disagreement_correct.values)
        drifts[stream] = [int(d) for d in
                          str(rows.drifts.iloc[0]).split(";") if d]
    if not correct:
        return (None, None), (None, None)
    return align(correct, drifts), align(diverse, drifts)


def _history_path(tag, stream, config, label_pct):
    return os.path.join(LAMBDA_HISTORY, tag, stream, config,
                        f"{label_pct}labels", f"history_seed{SEED}.csv")


def accuracy_aligned(windows, config, label_pct, streams, tag):
    """Aligned rolling accuracy, in percent."""
    values, drifts = {}, {}
    sub = windows[(windows.config == config) & (windows.label_pct == label_pct)]
    for stream in streams:
        path = _history_path(tag, stream, config, label_pct)
        if not os.path.exists(path):
            continue
        history = pd.read_csv(path)
        values[stream] = (history["instance"].values.astype(float),
                          history["rolling_acc"].values.astype(float) * 100)
        rows = sub[sub.dataset == stream]
        if rows.empty:
            continue
        drifts[stream] = [int(d) for d in
                          str(rows.drifts.iloc[0]).split(";") if d]
    values = {k: v for k, v in values.items() if k in drifts}
    if not values:
        return None, None
    return align(values, drifts)


def smooth(values, k=7):
    """Centred moving average that ignores gaps."""
    values = np.asarray(values, dtype=float)
    out = np.full_like(values, np.nan)
    half = k // 2
    for i in range(len(values)):
        window = values[max(0, i - half):i + half + 1]
        window = window[np.isfinite(window)]
        if len(window):
            out[i] = window.mean()
    return out
