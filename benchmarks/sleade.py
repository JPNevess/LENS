"""SLEADE, run from the authors' own implementation.

This row calls the SLEADE that ships with CapyMOA, driven by CapyMOA's own
prequential semi-supervised evaluator, so it runs the authors' code directly.
The ensemble size, the streams, the label rates, the seeds and the warm-up
window are the ones every other entry point uses, so the rows line up.

Mechanism
---------
SLEADE pairs a Streaming Random Patches ensemble with a disagreement-based use of
the unlabelled instances: the ensemble trains on whatever labels it is given, and
additionally exploits the unlabelled instances through the disagreement among its
members. Drift detection and member replacement are handled internally by SRP.

What is held fixed with the rest of the comparison
--------------------------------------------------
The ensemble size, the streams, the label rates, the seeds and the warm-up window
are the ones used by every other entry point here, so the rows line up. The base
learner is SRP's own, because replacing it would no longer be the authors'
method.

One detail is worth stating, because it is what makes the accuracies comparable
at all: CapyMOA's evaluator trains on the first ``INITIAL_LABELED`` instances but
does not score them, so a 100,000-instance stream reports 99,000. The evaluation
loop in ``lens/evaluation.py`` excludes the same warm-up window. Without that,
this row would be scored on an easier stream than the others.
"""
import argparse
import os
import sys
import time

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from lens._java import ensure_java_home

ensure_java_home()

from capymoa.evaluation import prequential_ssl_evaluation
from capymoa.ssl import SLEADE
from capymoa.stream import ARFFStream
from sklearn.metrics import f1_score

from lens.config import CONFIG_SLEADE, DIVERSITY_DISAGREEMENT
from lens.streams import resolve as resolve_stream

# --------------------------------------------------------------------- identity
CONFIG = CONFIG_SLEADE
NAME = "SLEADE"

# --------------------------------------------------------------------- protocol
ENSEMBLE_SIZE = 30       # members, matching every other method
INITIAL_LABELED = 1000   # fully labelled warm-up, trained on but not scored
WINDOW_SIZE = 1000       # evaluator's reporting window

RESULTS_CSV = os.path.join(_ROOT, "results", "benchmarks", "runs.csv")

DATASETS = ("AGR_a", "AGR_g", "RBF_m", "RBF_f", "LED_a", "LED_g",
            "airlines", "Electricity", "CovtFD")
LABEL_PCTS = (5, 1)

# Every reported cell is the mean over these five seeds, the same ones the other
# entry points use. A seed initialises the learner and, offset by
# LABEL_SEED_OFFSET, draws which instances arrive labelled.
SEEDS = (101, 102, 103, 104, 105)
LABEL_SEED_OFFSET = 1000

ROW_COLUMNS = [
    "dataset", "config", "method", "label_pct", "diversity_measure", "seed",
    "label_seed", "inference_mode", "training_mode", "global_acc", "f1_score",
    "drift_count", "total_instances", "elapsed_s", "error",
]


# ------------------------------------------------------------------------ runner
def dataset_path(name):
    """Resolve a stream name to a file, generating it if it is synthetic."""
    return resolve_stream(name)


def evaluate(path, label_pct, seed, max_instances=0):
    """One prequential semi-supervised run of the reference SLEADE."""
    stream = ARFFStream(path)
    learner = SLEADE(
        schema=stream.get_schema(),
        random_seed=seed,
        base_ensemble=f"StreamingRandomPatches -s {ENSEMBLE_SIZE}",
    )

    start = time.process_time()
    results = prequential_ssl_evaluation(
        stream=stream,
        learner=learner,
        label_probability=label_pct / 100.0,
        initial_window_size=INITIAL_LABELED,
        window_size=WINDOW_SIZE,
        max_instances=(max_instances if max_instances > 0 else None),
        random_seed=seed + LABEL_SEED_OFFSET,
        store_predictions=True,
        store_y=True,
    )
    elapsed = time.process_time() - start

    y_pred = list(results.predictions())
    y_true = list(results.ground_truth_y())
    # The first predictions can be None, before the ensemble has been trained.
    pairs = [(t, p) for t, p in zip(y_true, y_pred)
             if p is not None and t is not None]
    if pairs:
        truth, predicted = zip(*pairs)
        f1 = float(f1_score(truth, predicted, average="macro", zero_division=0))
    else:
        f1 = float("nan")

    return {
        "global_acc": results["cumulative"].accuracy() / 100.0,
        "f1_score": f1,
        "drift_count": -1,          # SLEADE does not expose its internal detector
        "total_instances": len(y_pred),
        "elapsed_s": elapsed,
        "error": None,
    }


def _row(dataset, label_pct, seed, result):
    return {
        "dataset": dataset,
        "config": CONFIG,
        "method": NAME,
        "label_pct": label_pct,
        "diversity_measure": DIVERSITY_DISAGREEMENT,
        "seed": seed,
        "label_seed": seed + LABEL_SEED_OFFSET,
        "inference_mode": "capymoa_sleade",
        "training_mode": "capymoa_sleade",
        "global_acc": result.get("global_acc"),
        "f1_score": result.get("f1_score"),
        "drift_count": result.get("drift_count"),
        "total_instances": result.get("total_instances"),
        "elapsed_s": result.get("elapsed_s"),
        "error": result.get("error"),
    }


def append_rows(path, rows):
    """Append the rows to the results CSV, replacing any run of the same cell."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows, columns=ROW_COLUMNS)
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
        df = df.drop_duplicates(
            subset=["dataset", "config", "label_pct", "seed"], keep="last")
    df.to_csv(path, index=False)


def report_cells(rows):
    """Print one line per reported cell: the mean over seeds, as in the table."""
    df = pd.DataFrame(rows, columns=ROW_COLUMNS)
    df = df[df.global_acc.notna()]
    if df.empty:
        return
    cells = df.groupby(["dataset", "label_pct"]).agg(
        acc=("global_acc", "mean"), f1=("f1_score", "mean"),
        n=("seed", "nunique")).reset_index()
    print(f"\n{NAME}: cell means over seeds")
    for _, c in cells.iterrows():
        note = "" if c.n > 1 else "   (one seed only, not comparable with the table)"
        print(f"  {c.dataset:<12s} {c.label_pct:>3d}% labels   "
              f"accuracy {100 * c.acc:5.2f}   macro-F1 {100 * c.f1:5.2f}   "
              f"{int(c.n)} seed(s){note}")


def build_parser():
    p = argparse.ArgumentParser(
        description=f"{NAME}: prequential semi-supervised evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS),
                   help="stream names to evaluate")
    p.add_argument("--label-pcts", type=int, nargs="+", default=list(LABEL_PCTS),
                   help="percentage of instances whose label is revealed")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS),
                   help="one run per seed; the seed selects the labelled subset")
    p.add_argument("--max-instances", type=int, default=0,
                   help="truncate each stream, for a quick check; 0 runs it whole")
    p.add_argument("--out", default=RESULTS_CSV, help="CSV to append rows to")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    rows = []
    for name in args.datasets:
        path = dataset_path(name)
        for label_pct in args.label_pcts:
            for seed in args.seeds:
                if not args.quiet:
                    print(f"{NAME:<12s} {name:<12s} {label_pct:>3d}% labels  "
                          f"seed {seed}", flush=True)
                result = evaluate(path, label_pct, seed, args.max_instances)
                rows.append(_row(name, label_pct, seed, result))
                if not args.quiet:
                    acc = result.get("global_acc")
                    print(f"{'':<12s} accuracy {100 * acc:.2f}%"
                          if acc is not None else f"{'':<12s} failed", flush=True)
    append_rows(args.out, rows)
    if not args.quiet:
        report_cells(rows)
        print(f"\n{len(rows)} run(s) written to {args.out}")
    return rows


if __name__ == "__main__":
    main()
