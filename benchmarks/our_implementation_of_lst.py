"""Self-training gated by meta-learned reliability rather than by own confidence.

The same self-training loop, with one change: what decides whether a member may
learn from its own prediction is the referee's estimate that the member is
correct, not the member's own margin. It separates being sure from being right,
which is exactly the failure mode of confidence-gated self-training.

Mechanism
---------
Gate and weight are both the referee estimate A-hat, scaled by label density.
A member that is confidently wrong is stopped by the gate, which the margin
cannot do.

Selection at prediction time
    none; all members vote with equal weight

Training on unlabelled instances
    each member trains on its own prediction when the referee's estimate clears the threshold, weighted by that estimate times label density

Relation to the published method
--------------------------------
Self-labelled techniques usually gate on a confidence measure derived from the
learner itself. Substituting a meta-learned reliability estimate is the point
of this column: it measures what the referee buys over raw confidence, holding
everything else fixed.

Every entry point in ``benchmarks/`` shares the evaluation harness of this
repository: the same base learners, the same ensemble size, the same streams and
the same prequential test-then-train protocol. That is deliberate. It means a
difference between two columns of the results table is a difference of
mechanism, not of implementation effort or of tuning.

Reference: Triguero et al., Self-labeled techniques for semi-supervised learning, Knowledge and Information Systems 42(2), 2015
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from lens.config import (DIVERSITY_DISAGREEMENT, CONFIG_13,
                         INF_NONE, TR_SELF_A)
from lens.ensemble import LENS
from lens.evaluation import run_experiment

# --------------------------------------------------------------------- identity
CONFIG = CONFIG_13
NAME = "LST*"
REFERENCE = "Triguero et al., Self-labeled techniques for semi-supervised learning, Knowledge and Information Systems 42(2), 2015"

INFERENCE_MODE = INF_NONE
TRAINING_MODE = TR_SELF_A

# --------------------------------------------------------------------- protocol
# Shared by every method in the comparison. Changing any of these here would make
# this column incomparable with the others.
ENSEMBLE_SIZE = 30       # members voting at any time
POOL_SIZE = 70           # background learners kept to replace weak members
GRACE_PERIOD = 50        # instances a leaf sees before a split is considered
TIE_THRESHOLD = 0.05     # Hoeffding tie-breaking threshold
CONFIDENCE = 0.01        # 1 - delta of the Hoeffding bound
LAMBDA_PARAM = 0.5       # initial relevance/diversity trade-off
DIVERSITY_MEASURE = DIVERSITY_DISAGREEMENT
UNSUPERVISED_DRIFT = False

DATA_DIR = os.path.join(_ROOT, "data")
RESULTS_CSV = os.path.join(_ROOT, "results", "benchmarks", "runs.csv")

DATASETS = ("AGR_a", "AGR_g", "RBF_m", "RBF_f", "LED_a", "LED_g",
            "airlines", "Electricity", "CovtFD")
LABEL_PCTS = (5, 1)

# Every reported cell is the mean over these five seeds. A seed fixes which
# instances are labelled and how the learners are initialised; a single seed can
# sit more than a point away from the mean on the noisier streams.
SEEDS = (42, 43, 44, 45, 46)

ROW_COLUMNS = [
    "dataset", "config", "method", "label_pct", "diversity_measure", "seed",
    "inference_mode", "training_mode", "global_acc", "f1_score", "drift_count",
    "total_instances", "elapsed_s", "error",
]


# -------------------------------------------------------------------- mechanism
class LabelledSelfTraining(LENS):
    """Uniform vote; the referee is used for admission, not for selection."""

    def competence(self, margins, est_acc):
        """Constant competence: selection is not part of this method."""
        return np.ones(self.ensemble_size)


# ------------------------------------------------------------------------ runner
def dataset_path(name):
    """Resolve a stream name to a file under ``data/``."""
    for ext in (".arff", ".csv"):
        path = os.path.join(DATA_DIR, name + ext)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"stream {name!r} not found in {DATA_DIR}. The synthetic streams and the "
        f"feature-drift Covertype variant are produced by "
        f"experiments/make_datasets.py.")


def evaluate(path, label_pct, seed, max_instances=0):
    """One prequential run of this method on one stream."""
    return run_experiment(
        dataset_path=path,
        config=CONFIG,
        label_pct=label_pct,
        seed=seed,
        inference_mode=INFERENCE_MODE,
        training_mode=TRAINING_MODE,
        ensemble_cls=LabelledSelfTraining,
        ensemble_size=ENSEMBLE_SIZE,
        pool_size=POOL_SIZE,
        grace_period=GRACE_PERIOD,
        tie_threshold=TIE_THRESHOLD,
        confidence=CONFIDENCE,
        lambda_param=LAMBDA_PARAM,
        diversity_measure=DIVERSITY_MEASURE,
        unsupervised_drift=UNSUPERVISED_DRIFT,
        max_instances=max_instances,
        verbose=False)


def _row(dataset, label_pct, seed, result):
    return {
        "dataset": dataset,
        "config": CONFIG,
        "method": NAME,
        "label_pct": label_pct,
        "diversity_measure": DIVERSITY_MEASURE,
        "seed": seed,
        "inference_mode": INFERENCE_MODE,
        "training_mode": TRAINING_MODE,
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
        description=f"{NAME} -- {REFERENCE}",
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
