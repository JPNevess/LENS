"""Prequential test-then-train evaluation of one configuration on one stream."""
import collections
import os
import time

import numpy as np

from ._java import ensure_java_home

ensure_java_home()

from capymoa.instance import LabeledInstance
from capymoa.stream import ARFFStream

from .baselines.sleade import _run_sleade_baseline
from .config import CONFIG_10, CONFIG_12, CONFIG_SLEADE, DIVERSITY_DISAGREEMENT
from .ensemble import LENS
from .history import DIAG_KEYS, _write_history_csv


def run_experiment(dataset_path, config, label_pct,
                   ensemble_size=30, grace_period=50, tie_threshold=0.05,
                   confidence=0.01, pool_size=70, pool_fresh_frac=0.5,
                   lambda_param=0.5,
                   diversity_measure=DIVERSITY_DISAGREEMENT,
                   subspace_frac=0.6, pseudo_conf_threshold=0.9,
                   pseudo_warmup_labels=200, div_batch_n=200,
                   adwin_delta=0.002, mlhat_alpha=0.05, adapt_lambda=True,
                   k_floor_frac=0.5, vote_temperature=0.5,
                   unsupervised_drift=False, self_train_referee=False,
                   inference_mode=None, training_mode=None, self_train_margin=0.5,
                   mmr_soft_weights=True, lambda_stable=0.85,
                   initial_labeled=1000,
                   max_instances=0, seed=42, verbose=True,
                   history_dir=None, diag_every=500, diag_all=False,
                   admission=False, admission_cap=150000,
                   referee_mode="mlhat", referee_probe=False,
                   referee_probe_cap=30000, referee_probe_preds=False,
                   ensemble_cls=None):
    """Run one configuration on one stream and return its metrics.

    ``ensemble_cls`` builds the ensemble; the entry points in ``benchmarks/`` pass
    their own subclass so the mechanism they implement is the one that runs.
    """
    if config == CONFIG_SLEADE:
        return _run_sleade_baseline(
            dataset_path     = dataset_path,
            label_pct        = label_pct,
            ensemble_size    = ensemble_size,
            diversity_measure= diversity_measure,
            initial_labeled  = initial_labeled,
            max_instances    = max_instances,
            seed             = seed,
            verbose          = verbose,
            history_dir      = history_dir,
            diag_every       = diag_every,
        )

    stream       = ARFFStream(dataset_path)
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

    ensemble = (ensemble_cls or LENS)(
        schema           = stream.get_schema(),
        config           = config,
        ensemble_size    = ensemble_size,
        lambda_param     = lambda_param,
        grace_period     = grace_period,
        tie_threshold    = tie_threshold,
        confidence       = confidence,
        pool_size        = pool_size,
        pool_fresh_frac  = pool_fresh_frac,
        diversity_measure= diversity_measure,
        subspace_frac        = subspace_frac,
        pseudo_conf_threshold= pseudo_conf_threshold,
        pseudo_warmup_labels = pseudo_warmup_labels,
        div_batch_n          = div_batch_n,
        adwin_delta          = adwin_delta,
        mlhat_alpha          = mlhat_alpha,
        adapt_lambda         = adapt_lambda,
        k_floor_frac         = k_floor_frac,
        vote_temperature     = vote_temperature,
        unsupervised_drift = unsupervised_drift,
        self_train_referee = self_train_referee,
        inference_mode   = inference_mode,
        training_mode    = training_mode,
        self_train_margin= self_train_margin,
        mmr_soft_weights = mmr_soft_weights,
        lambda_stable    = lambda_stable,
        warmup_labeled   = initial_labeled,
        referee_mode     = referee_mode,
        seed             = seed,
    )

    np.random.seed(seed)
    import random as _random
    _random.seed(seed)

    label_interval   = max(1, 100 // label_pct)
    rolling_correct  = collections.deque(maxlen=100)
    acc_hist         = collections.deque(maxlen=10000)
    global_correct   = 0
    total            = 0
    eval_total       = 0

    step_history     = []
    acc_history      = []
    k_history        = []
    lambda_history   = []
    y_true_all       = []
    y_pred_all       = []

    collect_diag = bool(history_dir and (config in (CONFIG_10, CONFIG_12) or diag_all))
    diag_step    = []
    diag_roll    = []
    diag         = {k: [] for k in DIAG_KEYS}

    adm_score_chunks, adm_correct_chunks, adm_name = [], [], [None]

    probe_estA_chunks, probe_trueC_chunks = [], []
    probe_preds_chunks = []
    probe_idx_chunks = []
    probe_margin_chunks = []

    start = time.process_time()

    while stream.has_more_instances():
        instance = stream.next_instance()
        y_true   = instance.y_index
        x_dict   = {f"feature_{i}": val for i, val in enumerate(instance.x)}

        pred = ensemble.predict(instance, x_dict)

        in_warmup = total < initial_labeled
        if referee_probe and not in_warmup:
            probe_estA_chunks.append(ensemble._last_pred_acc.astype(np.float32))
            probe_trueC_chunks.append(
                (ensemble._last_member_preds == y_true).astype(np.int8))
            probe_idx_chunks.append(total)
            probe_margin_chunks.append(
                ensemble._last_member_margins.astype(np.float32))
            if referee_probe_preds:
                probe_preds_chunks.append(
                    np.asarray(ensemble._last_member_preds, dtype=np.int16))

        if in_warmup or total % label_interval == 0:
            ensemble.train(instance, x_dict, y_true)
        else:
            if admission:
                sig = ensemble._admission_signals()
                if sig is not None:
                    scores, plabels, nm = sig
                    adm_name[0] = nm
                    ok = np.isfinite(scores)
                    if ok.any():
                        adm_score_chunks.append(scores[ok].astype(np.float32))
                        adm_correct_chunks.append(
                            (plabels[ok] == y_true).astype(np.int8))
            ensemble.train_unsupervised(instance, x_dict)

        total += 1
        is_correct = int(pred == y_true)
        rolling_correct.append(is_correct)
        acc_hist.append(is_correct)
        if not in_warmup:
            global_correct += is_correct
            eval_total     += 1

        if total % 5000 == 0 and len(acc_hist) >= 10000:
            h = np.fromiter(acc_hist, dtype=float)
            ensemble._acc_trend = float(h[-5000:].mean() - h[-10000:-5000].mean())

        if not in_warmup:
            y_true_all.append(y_true)
            y_pred_all.append(pred)

        if total % 100 == 0:
            step_history.append(total)
            acc_history.append(sum(rolling_correct) / len(rolling_correct))
            k_history.append(ensemble.current_k)
            lambda_history.append(ensemble.lambda_param)

        if collect_diag and total % diag_every == 0:
            snap = ensemble._snapshot_metrics()
            diag_step.append(total)
            diag_roll.append(sum(rolling_correct) / len(rolling_correct))
            for k in DIAG_KEYS:
                diag[k].append(snap[k])

        if verbose and total % 1000 == 0:
            print(f"  [{total}] config={config} label={label_pct}%  "
                  f"acc={acc_history[-1]:.4f}  K={ensemble.current_k}  "
                  f"λ={ensemble.lambda_param:.3f}")

        if max_instances > 0 and total >= max_instances:
            break

    elapsed    = time.process_time() - start
    global_acc = global_correct / eval_total if eval_total > 0 else 0.0

    from sklearn.metrics import f1_score as _f1
    f1 = float(_f1(y_true_all, y_pred_all, average="macro", zero_division=0)) if y_true_all else 0.0

    if collect_diag and diag_step:
        cols = {"instance": diag_step, "rolling_acc": diag_roll}
        cols.update({k: diag[k] for k in DIAG_KEYS})
        _write_history_csv(history_dir, dataset_name, config, label_pct, seed, cols)

    admission_out = None
    if admission and adm_score_chunks:
        adm_s = np.concatenate(adm_score_chunks)
        adm_c = np.concatenate(adm_correct_chunks)
        if admission_cap and adm_s.size > admission_cap:
            idx = np.random.RandomState(seed + 777).choice(
                adm_s.size, admission_cap, replace=False)
            adm_s, adm_c = adm_s[idx], adm_c[idx]
        admission_out = {"signal": adm_name[0],
                         "score": adm_s.astype(np.float32),
                         "correct": adm_c.astype(np.int8),
                         "n_total": int(sum(c.size for c in adm_score_chunks))}

    referee_probe_out = None
    if referee_probe and probe_estA_chunks:
        estA  = np.vstack(probe_estA_chunks)
        trueC = np.vstack(probe_trueC_chunks)
        preds = np.vstack(probe_preds_chunks) if probe_preds_chunks else None
        pidx  = np.asarray(probe_idx_chunks, dtype=np.int64)
        marg  = np.vstack(probe_margin_chunks)
        n_rows = estA.shape[0]
        if referee_probe_cap and n_rows > referee_probe_cap:
            idx = np.random.RandomState(seed + 999).choice(
                n_rows, referee_probe_cap, replace=False)
            idx.sort()
            estA, trueC, pidx, marg = estA[idx], trueC[idx], pidx[idx], marg[idx]
            if preds is not None:
                preds = preds[idx]
        referee_probe_out = {"referee_mode": ensemble.referee_mode,
                             "est_acc": estA.astype(np.float32),
                             "true_correct": trueC.astype(np.int8),
                             "member_preds": (preds.astype(np.int16)
                                              if preds is not None else None),
                             "instance_idx": pidx,
                             "margins": marg.astype(np.float32),
                             "n_total": int(n_rows)}

    return {
        "dataset"            : dataset_name,
        "config"             : config,
        "label_pct"          : label_pct,
        "diversity_measure"  : diversity_measure,
        "seed"               : seed,
        "inference_mode"     : ensemble.inference_mode,
        "training_mode"      : ensemble.training_mode,
        "use_mmr"            : ensemble.use_mmr,
        "use_pseudo_label"   : ensemble.use_pseudo_label,
        "unsup_drift"        : ensemble.use_unsup_drift,
        "inference_relevance": ensemble.inference_mode,
        "subspace_frac"        : subspace_frac,
        "ensemble_size"        : ensemble_size,
        "lambda_param"         : lambda_param,
        "adapt_lambda"         : adapt_lambda,
        "pool_size"            : pool_size,
        "adwin_delta"          : adwin_delta,
        "mlhat_alpha"          : mlhat_alpha,
        "pseudo_conf_threshold": pseudo_conf_threshold,
        "pseudo_warmup_labels" : pseudo_warmup_labels,
        "div_batch_n"          : div_batch_n,
        "global_acc"         : global_acc,
        "f1_score"           : f1,
        "drift_count"        : ensemble.drift_count,
        "total_instances"    : eval_total,
        "elapsed_s"          : elapsed,
        "lbd_updates"        : int(ensemble._lbd_member_updates),
        "lbd_instances"      : int(ensemble._lbd_n_instances),
        "lbd_gap_mean"       : (ensemble._lbd_gap_sum / ensemble._lbd_member_updates
                                if ensemble._lbd_member_updates else float("nan")),
        "lbd_conf_mean"      : (ensemble._lbd_conf_sum / ensemble._lbd_member_updates
                                if ensemble._lbd_member_updates else float("nan")),
        "lbd_gap_pre_mean"   : (ensemble._lbd_gap_pre_sum / ensemble._lbd_pre_count
                                if ensemble._lbd_pre_count else float("nan")),
        "lbd_conf_pre_mean"  : (ensemble._lbd_conf_pre_sum / ensemble._lbd_pre_count
                                if ensemble._lbd_pre_count else float("nan")),
        "lbd_pool_updates"   : int(ensemble._lbd_pool_updates),
        "step_history"       : step_history,
        "acc_history"        : acc_history,
        "k_history"          : k_history,
        "lambda_history"     : lambda_history,
        "y_true"             : y_true_all,
        "y_pred"             : y_pred_all,
        "admission"          : admission_out,
        "referee_probe"      : referee_probe_out,
        "referee_mode"       : ensemble.referee_mode,
    }
