"""Selection by referee-estimated accuracy.

Competence comes from the referee, the multi-target meta-learner that predicts
per instance whether each member is currently correct. Unlike the margin, this
is a learned signal: it can tell that a member is confidently wrong in this
region of the space, which no confidence measure can.

The estimate is available without the true label, so it can be used at
prediction time on every instance rather than only on the labelled ones.

Selection at prediction time
    members ranked by the referee's estimated accuracy A-hat; the top-K vote

Training on unlabelled instances
    none; unlabelled instances are discarded

Relation to the published method
--------------------------------
META-DES picks competent members from a validation neighbourhood of the query,
using several meta-features. This row replaces the neighbourhood and the
meta-features with an online meta-learner that generalises across the stream,
because a stored validation set does not survive concept drift.

It is not a port of META-DES. It is this repository's ensemble ranking members
by the referee's estimate, with self-training off.

Every entry point in ``benchmarks/`` shares the evaluation harness of this
repository: the same base learners, the same ensemble size, the same streams
and the same prequential test-then-train protocol. That is deliberate. It means
a difference between two rows of the results table is a difference of
mechanism, not of implementation effort or of tuning.

Reference: Cruz et al., META-DES: a dynamic ensemble selection framework using meta-learning, Pattern Recognition 48(5), 2015
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from lens.config import (DIVERSITY_DISAGREEMENT, CONFIG_2,
                         INF_MLHAT_A, TR_NONE)
from lens.ensemble import LENS
from lens.evaluation import run_experiment
from lens.streams import resolve as resolve_stream

# --------------------------------------------------------------------- identity
CONFIG = CONFIG_2
NAME = "Meta-BR"

INFERENCE_MODE = INF_MLHAT_A
TRAINING_MODE = TR_NONE

# --------------------------------------------------------------------- protocol
# Shared by every entry point in this directory. Changing any of these here would
# make this row incomparable with the others.
ENSEMBLE_SIZE = 30       # members voting at any time
POOL_SIZE = 70           # background learners kept to replace weak members
GRACE_PERIOD = 50        # instances a leaf sees before a split is considered
TIE_THRESHOLD = 0.05     # Hoeffding tie-breaking threshold
CONFIDENCE = 0.01        # 1 - delta of the Hoeffding bound
LAMBDA_PARAM = 0.5       # initial relevance/diversity trade-off
DIVERSITY_MEASURE = DIVERSITY_DISAGREEMENT
UNSUPERVISED_DRIFT = False

RESULTS_CSV = os.path.join(_ROOT, "results", "benchmarks", "runs.csv")

# The synthetic streams are generated from lens/streams.py on first use and
# cached under data/; the real ones have to be present there already.
DATASETS = ("AGR_a", "AGR_g", "RBF_m", "RBF_f", "LED_a", "LED_g",
            "airlines", "Electricity", "CovtFD")
LABEL_PCTS = (5, 1)

# Every reported cell is the mean over these five seeds. A seed initialises the
# learners -- feature subspaces, tie-breaking, the background pool -- and, offset
# by LABEL_SEED_OFFSET, also draws which instances arrive labelled. A single seed
# can sit more than a point away from the mean on the noisier streams, so one
# seed is not comparable with a reported cell.
SEEDS = (101, 217, 349, 523, 811)

# Keeps the labelled subset from being a deterministic function of the learner
# initialisation, so the two can be varied independently.
LABEL_SEED_OFFSET = 4703

ROW_COLUMNS = [
    "dataset", "config", "method", "label_pct", "diversity_measure", "seed",
    "label_seed", "inference_mode", "training_mode", "global_acc", "f1_score",
    "drift_count", "total_instances", "elapsed_s", "error",
]


# -------------------------------------------------------------------- mechanism
class MetaBinaryRelevance(LENS):
    """Competence is the referee's estimate that the member is correct here."""

    def competence(self, margins, est_acc):
        """The referee's per-member probability of being correct, which needs no label at prediction time."""
        return est_acc


# ------------------------------------------------------------------------ runner
def dataset_path(name):
    """Resolve a stream name to a file, generating it if it is synthetic."""
    return resolve_stream(name)


def evaluate(path, label_pct, seed, max_instances=0):
    """One prequential run of this configuration on one stream."""
    return run_experiment(
        dataset_path=path,
        config=CONFIG,
        label_pct=label_pct,
        seed=seed,
        label_seed=seed + LABEL_SEED_OFFSET,
        inference_mode=INFERENCE_MODE,
        training_mode=TRAINING_MODE,
        ensemble_cls=MetaBinaryRelevance,
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
        "label_seed": seed + LABEL_SEED_OFFSET,
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
        description=f"{NAME} -- selection by referee-estimated accuracy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS),
                   help="stream names to evaluate")
    p.add_argument("--label-pcts", type=int, nargs="+", default=list(LABEL_PCTS),
                   help="percentage of instances whose label is revealed")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS),
                   help="one run per seed; the seed initialises the learners and "
                        "selects the labelled subset")
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
