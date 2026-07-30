"""SLEADE baseline.

Not part of LENS: this runs the reference SLEADE implementation shipped with
CapyMOA under the same prequential semi-supervised protocol, so the comparison
in the paper uses the authors' own code.
"""
import os
import time

import numpy as np

from .._java import ensure_java_home

ensure_java_home()

from capymoa.stream import ARFFStream

from ..config import CONFIG_SLEADE
from ..history import _rolling_accuracy, _write_history_csv


def _run_sleade_baseline(dataset_path, label_pct, ensemble_size,
                         diversity_measure, max_instances, seed, verbose=True,
                         initial_labeled=1000, history_dir=None, diag_every=500):
    """Run the reference SLEADE implementation from CapyMOA under prequential
    semi-supervised evaluation.
    """
    from capymoa.ssl import SLEADE
    from capymoa.evaluation import prequential_ssl_evaluation
    from sklearn.metrics import f1_score as _f1

    stream       = ARFFStream(dataset_path)
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

    learner = SLEADE(
        schema        = stream.get_schema(),
        random_seed   = seed,
        base_ensemble = f"StreamingRandomPatches -s {ensemble_size}",
    )

    if verbose:
        print(f"  [SLEADE/CapyMOA] {dataset_name}  label={label_pct}%  "
              f"ensemble={ensemble_size}")

    start   = time.process_time()
    results = prequential_ssl_evaluation(
        stream             = stream,
        learner            = learner,
        label_probability  = label_pct / 100.0,
        initial_window_size= initial_labeled,
        window_size        = 1000,
        max_instances      = (max_instances if max_instances > 0 else None),
        random_seed        = seed,
        store_predictions  = True,
        store_y            = True,
    )
    elapsed = time.process_time() - start

    cum        = results["cumulative"]
    global_acc = cum.accuracy() / 100.0

    y_pred = list(results.predictions())
    y_true = list(results.ground_truth_y())
    pairs  = [(t, p) for t, p in zip(y_true, y_pred) if p is not None and t is not None]
    if pairs:
        yt, yp = zip(*pairs)
        f1 = float(_f1(yt, yp, average="macro", zero_division=0))
    else:
        f1 = float("nan")
    total = len(y_pred)

    if history_dir:
        steps, accs = _rolling_accuracy(y_true, y_pred, window=100, every=diag_every)
        _write_history_csv(history_dir, dataset_name, CONFIG_SLEADE, label_pct, seed,
                           {"instance": steps, "rolling_acc": accs})

    return {
        "dataset"            : dataset_name,
        "config"             : CONFIG_SLEADE,
        "label_pct"          : label_pct,
        "diversity_measure"  : diversity_measure,
        "seed"               : seed,
        "use_mmr"            : False,
        "use_pseudo_label"   : True,
        "unsup_drift"        : False,
        "inference_relevance": "capymoa_sleade",
        "global_acc"         : global_acc,
        "f1_score"           : f1,
        "drift_count"        : -1,
        "total_instances"    : total,
        "elapsed_s"          : elapsed,
        "step_history"       : [],
        "acc_history"        : [],
        "k_history"          : [],
        "lambda_history"     : [],
        "y_true"             : y_true,
        "y_pred"             : y_pred,
    }
