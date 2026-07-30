"""Per-run time series written alongside the metrics, used by the figures that
show behaviour over the stream rather than a single aggregate.
"""
import collections
import os

import numpy as np
import pandas as pd


DIAG_KEYS = (
    "active_real_acc", "active_pred_acc", "active_real_div", "active_pred_div",
    "pool_real_acc", "pool_real_div", "K", "lambda", "drift_count", "n_swapped",
    "mean_margin", "mean_A_inst", "mean_C", "mean_w",
    "lbd_updates", "lbd_instances", "lbd_gap_mean", "lbd_conf_mean",
    "lbd_gap_pre_mean", "lbd_conf_pre_mean",
    "lbd_pool_updates",
)


def _history_path(history_dir, dataset, config, label_pct, seed):
    """Path of the per-run history CSV."""
    out_dir = os.path.join(history_dir, dataset, config, f"{label_pct}labels")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"history_seed{seed}.csv")


def _write_history_csv(history_dir, dataset, config, label_pct, seed, columns):
    """Write a history CSV from a dict of columns."""
    path = _history_path(history_dir, dataset, config, label_pct, seed)
    pd.DataFrame(columns).to_csv(path, index=False)
    return path


def _rolling_accuracy(y_true, y_pred, window=100, every=500):
    """Prequential accuracy in a sliding window, sampled every ``every`` instances."""
    import collections as _c
    win   = _c.deque(maxlen=window)
    steps, accs = [], []
    n = 0
    for t, p in zip(y_true, y_pred):
        if p is None or t is None:
            continue
        n += 1
        win.append(1 if int(p) == int(t) else 0)
        if n % every == 0:
            steps.append(n)
            accs.append(sum(win) / len(win))
    return steps, accs

