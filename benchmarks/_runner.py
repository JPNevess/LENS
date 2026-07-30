"""Shared protocol for the benchmark entry points.

Every method in the comparison is evaluated the same way: prequential
test-then-train on the same streams, with the same ensemble size, base learner
settings and label rates. Only the selection and self-training mechanisms differ,
which is what the comparison is meant to isolate. Keeping the protocol in one
place is what makes that guarantee checkable.
"""
import argparse
import os
import sys
from collections import namedtuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lens.config import DIVERSITY_DISAGREEMENT
from lens.evaluation import run_experiment

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")
RESULTS_CSV = os.path.join(_ROOT, "results", "benchmarks", "runs.csv")

# Evaluation protocol shared by every method.
PROTOCOL = dict(
    ensemble_size=30,
    pool_size=70,
    grace_period=50,
    tie_threshold=0.05,
    confidence=0.01,
    lambda_param=0.5,
    diversity_measure=DIVERSITY_DISAGREEMENT,
    unsupervised_drift=False,
)

# Streams used in the paper. The large one is generated locally rather than
# shipped; see data/README.md.
DATASETS = ("AGR_a", "AGR_g", "RBF_m", "RBF_f", "LED_a", "LED_g",
            "airlines", "Electricity", "CovtFD")
LABEL_PCTS = (5, 1)

Method = namedtuple("Method", "key name reference inference_mode training_mode")

ROW_COLUMNS = [
    "dataset", "config", "method", "label_pct", "diversity_measure", "seed",
    "inference_mode", "training_mode", "global_acc", "f1_score", "drift_count",
    "total_instances", "elapsed_s", "error",
]


def dataset_path(name):
    """Resolve a stream name to a file under ``data/``."""
    for ext in (".arff", ".csv"):
        path = os.path.join(DATA_DIR, name + ext)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"stream {name!r} not found in {DATA_DIR}. Synthetic streams and the "
        f"feature-drift Covertype variant are produced by "
        f"experiments/make_datasets.py.")


def build_parser(method):
    p = argparse.ArgumentParser(
        description=f"{method.name} -- {method.reference}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS),
                   help="stream names to evaluate")
    p.add_argument("--label-pcts", type=int, nargs="+", default=list(LABEL_PCTS),
                   help="percentage of instances whose label is revealed")
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                   help="one run per seed; the seed selects the labelled subset")
    p.add_argument("--max-instances", type=int, default=None,
                   help="truncate each stream, for a quick check")
    p.add_argument("--out", default=RESULTS_CSV, help="CSV to append rows to")
    p.add_argument("--quiet", action="store_true")
    return p


def _row(method, dataset, label_pct, seed, result):
    return {
        "dataset": dataset,
        "config": method.key,
        "method": method.name,
        "label_pct": label_pct,
        "diversity_measure": PROTOCOL["diversity_measure"],
        "seed": seed,
        "inference_mode": method.inference_mode,
        "training_mode": method.training_mode,
        "global_acc": result.get("global_acc"),
        "f1_score": result.get("f1_score"),
        "drift_count": result.get("drift_count"),
        "total_instances": result.get("total_instances"),
        "elapsed_s": result.get("elapsed_s"),
        "error": result.get("error"),
    }


def _append(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows, columns=ROW_COLUMNS)
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
        df = df.drop_duplicates(
            subset=["dataset", "config", "label_pct", "seed"], keep="last")
    df.to_csv(path, index=False)


def main(method, argv=None):
    """Evaluate one method over the requested streams and append the rows."""
    args = build_parser(method).parse_args(argv)
    rows = []
    for name in args.datasets:
        path = dataset_path(name)
        for label_pct in args.label_pcts:
            for seed in args.seeds:
                if not args.quiet:
                    print(f"{method.name:<12s} {name:<12s} "
                          f"{label_pct:>3d}% labels  seed {seed}", flush=True)
                result = run_experiment(
                    dataset_path=path,
                    config=method.key,
                    label_pct=label_pct,
                    seed=seed,
                    inference_mode=method.inference_mode,
                    training_mode=method.training_mode,
                    max_instances=args.max_instances,
                    verbose=False,
                    **PROTOCOL)
                rows.append(_row(method, name, label_pct, seed, result))
                if not args.quiet:
                    acc = result.get("global_acc")
                    print(f"{'':<12s} accuracy {100 * acc:.2f}%"
                          if acc is not None else f"{'':<12s} failed", flush=True)
    _append(args.out, rows)
    if not args.quiet:
        print(f"\n{len(rows)} run(s) written to {args.out}")
    return rows
