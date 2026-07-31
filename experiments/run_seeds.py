"""Repeat the grid over several seeds, to get a spread per cell.

The seed selects which instances have their label revealed, so repeating it
measures how much of the difference between methods is due to that draw. Results
go to results/seeds/runs.csv and are merged with the single-seed grid by the
figures.

    python experiments/run_seeds.py --seeds 101 102 103 104 105
"""

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from lens.config import (
    CONFIG_SLEADE as CONFIG_0,
    DIVERSITY_DISAGREEMENT,
    INF_MMR,
    PAPER_CONFIGS,
)
from lens import streams
from lens.evaluation import run_experiment

MMR_CONFIGS = tuple(c for c, (inf, _) in PAPER_CONFIGS.items() if inf == INF_MMR)


OUTPUT_DIR    = os.path.join(_ROOT, "results", "seeds")
CSV_NAME      = "runs.csv"

CONFIGS       = (CONFIG_0,) + tuple(PAPER_CONFIGS.keys())
LABEL_PCTS    = (5, 1)
SEEDS         = (101, 102, 103, 104, 105)
ENSEMBLE_SIZE = 30
GRACE_PERIOD  = 50
TIE_THRESHOLD = 0.05
CONFIDENCE    = 0.01
POOL_SIZE     = 70
LAMBDA_PARAM  = 0.5
DIVERSITY     = DIVERSITY_DISAGREEMENT
MAX_WORKERS   = 2



def _worker(kwargs):
    ds = os.path.splitext(os.path.basename(kwargs["dataset_path"]))[0]
    print(f"  >> START {ds:18s} {kwargs['config']}  label={kwargs['label_pct']:3d}%  "
          f"seed={kwargs['seed']}  (pid={os.getpid()})", flush=True)
    return run_experiment(**kwargs)


def run_parallel_incertezas(dataset_paths, configs=CONFIGS, label_pcts=LABEL_PCTS,
                            seeds=SEEDS, ensemble_size=ENSEMBLE_SIZE,
                            grace_period=GRACE_PERIOD, tie_threshold=TIE_THRESHOLD,
                            confidence=CONFIDENCE, pool_size=POOL_SIZE,
                            lambda_param=LAMBDA_PARAM, diversity=DIVERSITY,
                            max_instances=0, output_dir=OUTPUT_DIR,
                            workers=MAX_WORKERS):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, CSV_NAME)

    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        if "global_acc" in existing_df.columns:
            n_err = int(existing_df["global_acc"].isna().sum())
            if n_err:
                print(f"  [cleanup] {n_err} stale error rows dropped "
                      f"from the CSV, so they will run again).")
                existing_df = existing_df[existing_df["global_acc"].notna()]
        all_rows  = existing_df.to_dict("records")
        completed = {
            (str(r["dataset"]), str(r["config"]), int(r["label_pct"]),
             int(r["seed"]))
            for r in all_rows
            if not (isinstance(r.get("global_acc"), float) and pd.isna(r.get("global_acc")))
        }
        print(f"  [resume] {len(completed)} runs already in the CSV, skipping them.")
    else:
        all_rows  = []
        completed = set()

    tasks   = []
    skipped = 0
    for dataset_path in dataset_paths:
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        for config in configs:
            for label_pct in label_pcts:
                for seed in seeds:
                    key = (dataset_name, config, label_pct, seed)
                    if key in completed:
                        skipped += 1
                        continue
                    tasks.append(dict(
                        dataset_path     = dataset_path,
                        config           = config,
                        label_pct        = label_pct,
                        seed             = seed,
                        ensemble_size    = ensemble_size,
                        grace_period     = grace_period,
                        tie_threshold    = tie_threshold,
                        confidence       = confidence,
                        pool_size        = pool_size,
                        lambda_param     = lambda_param,
                        diversity_measure= diversity,
                        unsupervised_drift = False,
                        max_instances    = max_instances,
                        verbose          = False,
                        history_dir      = None,
                    ))

    n_todo = len(tasks)
    print(f"\n{'='*65}")
    print(f"  {n_todo} runs a executar  |  {skipped} saltadas (resume)  |  {workers} workers")
    print(f"{'='*65}")

    if not tasks:
        print("  -> every run already completed, nothing to do.")
        return pd.DataFrame(all_rows)

    done  = 0
    start = time.time()

    import multiprocessing as _mp
    from concurrent.futures.process import BrokenProcessPool

    MAX_RETRY_ROUNDS = 3
    pending = tasks
    round_n = 0
    while pending:
        round_n += 1
        if round_n > 1:
            print(f"\n  [RETRY ronda {round_n}] a re-tentar {len(pending)} "
                  f"tarefas falhadas com um pool novo...")
        retry_next = []
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=_mp.get_context("spawn"),
                                 max_tasks_per_child=1) as executor:
            future_to_task = {executor.submit(_worker, t): t for t in pending}

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                ds   = os.path.splitext(os.path.basename(task["dataset_path"]))[0]
                tag  = (f"{ds:18s} {task['config']}  "
                        f"label={task['label_pct']:3d}%  seed={task['seed']}")

                try:
                    result = future.result()
                except Exception as exc:
                    transient = (isinstance(exc, BrokenProcessPool)
                                 or "Java Virtual Machine" in str(exc))
                    if transient and round_n < MAX_RETRY_ROUNDS:
                        print(f"  RETRY {tag}  -> {type(exc).__name__}: {exc}")
                        retry_next.append(task)
                    else:
                        done += 1
                        print(f"  ERR [{done}/{n_todo}] {tag}  -> {exc}")
                        all_rows.append({
                            "dataset"            : ds,
                            "config"             : task["config"],
                            "label_pct"          : task["label_pct"],
                            "diversity_measure"  : task["diversity_measure"],
                            "seed"               : task["seed"],
                            "use_mmr"            : False,
                            "use_pseudo_label"   : False,
                            "unsup_drift"        : False,
                            "inference_relevance": "unknown",
                            "global_acc"         : float("nan"),
                            "f1_score"           : float("nan"),
                            "drift_count"        : -1,
                            "total_instances"    : 0,
                            "elapsed_s"          : 0.0,
                            "error"              : str(exc),
                        })
                        pd.DataFrame(all_rows).to_csv(csv_path, index=False)
                    continue

                done += 1
                all_rows.append({
                    "dataset"            : result["dataset"],
                    "config"             : result["config"],
                    "label_pct"          : result["label_pct"],
                    "diversity_measure"  : result["diversity_measure"],
                    "seed"               : result["seed"],
                    "inference_mode"     : result.get("inference_mode", "capymoa"),
                    "training_mode"      : result.get("training_mode", "capymoa"),
                    "use_mmr"            : result["use_mmr"],
                    "use_pseudo_label"   : result["use_pseudo_label"],
                    "unsup_drift"        : result.get("unsup_drift", False),
                    "inference_relevance": result["inference_relevance"],
                    "global_acc"         : round(result["global_acc"], 4),
                    "f1_score"           : round(result["f1_score"],   4),
                    "drift_count"        : result["drift_count"],
                    "total_instances"    : result["total_instances"],
                    "elapsed_s"          : round(result["elapsed_s"],  1),
                    "lbd_updates"        : result.get("lbd_updates", 0),
                    "lbd_instances"      : result.get("lbd_instances", 0),
                    "lbd_gap_mean"       : result.get("lbd_gap_mean", float("nan")),
                    "lbd_conf_mean"      : result.get("lbd_conf_mean", float("nan")),
                    "lbd_gap_pre_mean"   : result.get("lbd_gap_pre_mean", float("nan")),
                    "lbd_conf_pre_mean"  : result.get("lbd_conf_pre_mean", float("nan")),
                    "lbd_pool_updates"   : result.get("lbd_pool_updates", 0),
                })
                print(f"  OK  [{done}/{n_todo}] {tag}  "
                      f"acc={result['global_acc']:.4f}  "
                      f"f1={result['f1_score']:.4f}  "
                      f"drifts={result['drift_count']}  "
                      f"t={result['elapsed_s']:.1f}s")

                pd.DataFrame(all_rows).to_csv(csv_path, index=False)

        pending = retry_next

    print(f"\nDone. {done} new run(s) in {time.time()-start:.0f}s. "
          f"{len(all_rows)} rows in the CSV.\nResults in {csv_path}")

    return pd.DataFrame(all_rows)


def _print_summary(df):
    if df.empty or "global_acc" not in df.columns:
        return
    print("\n" + "=" * 65)
    print("  summary: mean +- std over seeds per (stream, config, label rate)")
    print("=" * 65)
    grp = df.groupby(["dataset", "config", "label_pct"])[
        ["global_acc", "f1_score", "drift_count"]]
    summary = grp.agg(["mean", "std"]).round(4)
    print(summary.to_string())

    std = df.groupby(["dataset", "config", "label_pct"])["global_acc"].std().dropna()
    if not std.empty:
        print("\n" + "=" * 65)
        print("  Variability: standard deviation of accuracy over seeds, per configuration")
        print("=" * 65)
        agg = std.groupby("config").agg(["mean", "max"]).round(4)
        agg.columns = ["std_media", "std_max"]
        print(agg.sort_values("std_media", ascending=False).to_string())
    print("=" * 65)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="MMR-DEMS incertezas runner (3 seeds, paralelo)")
    parser.add_argument("--quick", action="store_true",
                        help="cap each run at 5000 instances, for a quick check")
    parser.add_argument("--max-instances", type=int, default=None, dest="max_instances",
                        help="cap on instances per run (0 or omitted means the whole stream, "
                             "--quick = 5000).")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="subset of the streams (e.g. Electricity LED_a)")
    parser.add_argument("--configs", type=str, nargs="+", default=list(CONFIGS),
                        help="subset of the configurations (e.g. config_12)")
    parser.add_argument("--label-pcts", type=int, nargs="+", default=list(LABEL_PCTS),
                        dest="label_pcts")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS),
                        help=f"seeds to run (default: {list(SEEDS)})")
    parser.add_argument("--include-big", dest="include_big", action="store_true",
                        help="include the large streams (CovtFD, ForestCoverType, "
                             "PokerHand), which are excluded by default because "
                             "each run takes hours")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"number of parallel worker processes (default: {MAX_WORKERS})")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = _parse_args()

    dataset_paths = streams.resolve_all(
        args.datasets, include_big=bool(args.datasets or args.include_big))

    if not dataset_paths:
        print("no streams resolved; nothing to run")
        return

    dataset_paths = sorted(dataset_paths, key=os.path.getsize)

    seeds = tuple(args.seeds)

    if args.max_instances is not None:
        max_instances = args.max_instances
    elif args.quick:
        max_instances = 5000
    else:
        max_instances = 0

    configs = tuple(args.configs)

    print("=" * 65)
    print("  MMR-DEMS Incertezas (multi-seed, paralelo)")
    print("=" * 65)
    print(f"  Datasets    : {[os.path.basename(p) for p in dataset_paths]}")
    print(f"  Configs     : {list(configs)}")
    print(f"  Label pcts  : {args.label_pcts}")
    print(f"  Seeds       : {seeds}  ({len(seeds)} runs per configuration and stream)")
    print(f"  Workers     : {args.workers}")
    print(f"  Max inst.   : {max_instances if max_instances else 'all'}")
    print(f"  Output      : {os.path.join(args.output, CSV_NAME)}")
    print("=" * 65)

    df = run_parallel_incertezas(
        dataset_paths    = dataset_paths,
        label_pcts       = tuple(args.label_pcts),
        configs          = configs,
        seeds            = seeds,
        ensemble_size    = ENSEMBLE_SIZE,
        grace_period     = GRACE_PERIOD,
        tie_threshold    = TIE_THRESHOLD,
        confidence       = CONFIDENCE,
        pool_size        = POOL_SIZE,
        lambda_param     = LAMBDA_PARAM,
        diversity        = DIVERSITY,
        max_instances    = max_instances,
        output_dir       = args.output,
        workers          = args.workers,
    )

    _print_summary(df)


if __name__ == "__main__":
    main()
