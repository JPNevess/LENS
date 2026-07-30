"""Loader for the benchmark comparison, shared by Table 1 and Figure 4.

Both read the same cell means so they cannot disagree with each other: a cell is
one (stream, method, label rate), averaged over the seeds that ran for it.

Reads: results/ablation/runs.csv and results/seeds/runs.csv
"""
import numpy as np
import pandas as pd

from _common import BENCHMARK_CSV, SEEDS_CSV

# Column order of the table, which is also the order of the methods in Figure 4.
METHODS = [
    ("config_base", "ARF"),
    ("config_1", "DyAbst"),
    ("config_2", r"Meta$_{BR}$"),
    ("config_3", "DEMS"),
    ("config_4", "DynED"),
    ("config_5", "Self-train"),
    ("config_13", "LST*"),
    ("config_6", "SCo-For"),
    ("config_0", "SLEADE"),
    ("config_10", r"LENS$^M$"),
    ("config_12", "LENS"),
]
CONFIGS = [c for c, _ in METHODS]
PLAIN_NAMES = {"config_base": "ARF", "config_1": "DyAbst", "config_2": "Meta-BR",
               "config_3": "DEMS", "config_4": "DynED", "config_5": "Self-train",
               "config_13": "LST*", "config_6": "SCo-For", "config_0": "SLEADE",
               "config_10": "LENS-M", "config_12": "LENS"}
PROPOSED = "config_12"

# Row order of the table.
STREAMS = [
    (r"LED$_a$", "LED_a"), (r"LED$_g$", "LED_g"),
    (r"AGR$_a$", "AGR_a"), (r"AGR$_g$", "AGR_g"),
    (r"RBF$_m$", "RBF_m"), (r"RBF$_f$", "RBF_f"),
    ("AIRL", "airlines"), ("ELEC", "Electricity"), ("COVT", "CovtFD"),
]
LABEL_PCTS = (1, 5)

_COLUMNS = ["dataset", "config", "label_pct", "seed", "global_acc", "f1_score"]


def load():
    """Cell means and standard deviations over seeds, one row per cell."""
    frames = [pd.read_csv(BENCHMARK_CSV)[_COLUMNS],
              pd.read_csv(SEEDS_CSV)[_COLUMNS]]
    df = pd.concat(frames, ignore_index=True)
    df = df[df.global_acc.notna()]
    # A run may appear in both tables; the first occurrence wins.
    df = df.drop_duplicates(["dataset", "config", "label_pct", "seed"])
    return df.groupby(["dataset", "config", "label_pct"]).agg(
        acc_m=("global_acc", "mean"), acc_s=("global_acc", "std"),
        f1_m=("f1_score", "mean"), f1_s=("f1_score", "std"),
        n=("seed", "nunique")).reset_index()


def cell_matrix(cells, metric, label_pct):
    """Methods by streams matrix of cell means, in table order."""
    column = f"{metric}_m"
    out = np.full((len(CONFIGS), len(STREAMS)), np.nan)
    for i, config in enumerate(CONFIGS):
        for j, (_, stream) in enumerate(STREAMS):
            row = cells[(cells.dataset == stream) & (cells.config == config)
                        & (cells.label_pct == label_pct)]
            if len(row):
                out[i, j] = float(row[column].iloc[0])
    return out


def average_ranks(matrix):
    """Mean rank per method over streams, where rank 1 is the best."""
    ranks = np.vstack([pd.Series(matrix[:, j]).rank(ascending=False).values
                       for j in range(matrix.shape[1])]).T
    return ranks.mean(axis=1)
