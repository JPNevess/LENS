"""Selection and self-training modes, and the identifiers used in results files.

The ensemble is parametrised along two orthogonal axes:

* ``inference_mode`` -- how member votes are weighted and selected at prediction
  time (dynamic ensemble selection).
* ``training_mode`` -- how unlabelled instances are turned into pseudo-labels and
  weighted during training (semi-supervised learning).

Each entry point in ``benchmarks/`` is one point in this space. The ``config_*``
identifiers below are the keys under which every run is stored in ``results/``,
so they are kept stable; ``METHOD_NAMES`` maps them to the mechanism names used
in the paper.

Every configuration here is a cell of this repository's own factorial ablation.
None of them is a port of a method from the literature, and none should be
presented as one. Mechanisms are named after what they compute -- margin,
referee accuracy, maximal marginal relevance -- and where a cell isolates the
idea a published method is built around, the corresponding file in
``benchmarks/`` says so in prose without claiming to reproduce that method's
results. ``CONFIG_SLEADE`` is the one exception: it does not go through this
package at all, it runs the reference implementation shipped with CapyMOA.
"""

# Inference (dynamic ensemble selection).
INF_NONE = "none"          # uniform majority vote
INF_MARGIN = "margin"      # competence = prediction margin M
INF_MLHAT_A = "mlhat_A"    # competence = meta-learner estimated accuracy A-hat
INF_SQRTAM = "sqrtAM"      # competence = sqrt(A-hat * M)
INF_MMR = "mmr"            # sqrt(A-hat * M) plus maximal marginal relevance

# Training (semi-supervised learning).
TR_NONE = "none"                  # labelled instances only
TR_SELF_M = "self_train_M"        # self-train on own prediction when margin is high
TR_SELF_A = "self_train_A"        # self-train gated and weighted by A-hat only
TR_DISAGREE_C = "disagree_c"      # learn-by-disagreement, weight = consensus confidence
TR_TRAIN_C = "train_C"            # learn-by-disagreement, weight = sqrt(A-hat * M)
TR_TRAIN_W = "train_w"            # learn-by-disagreement, weight = sqrt(A-bar * c)

# Diversity measures available for the MMR redundancy term.
DIVERSITY_DISAGREEMENT = "disagreement"
DIVERSITY_Q_STATISTIC = "q_statistic"
DIVERSITY_KAPPA = "kappa"
DIVERSITY_DOUBLE_FAULT = "double_fault"
DIVERSITY_CORRELATION = "correlation"

# SLEADE is an external baseline: it does not go through the ensemble in this
# package, it is run through the reference implementation in
# ``lens.baselines.sleade``.
CONFIG_SLEADE = "config_0"

CONFIG_BASE = "config_base"
CONFIG_1 = "config_1"
CONFIG_2 = "config_2"
CONFIG_3 = "config_3"
CONFIG_4 = "config_4"
CONFIG_5 = "config_5"
CONFIG_6 = "config_6"
CONFIG_7 = "config_7"
CONFIG_8 = "config_8"
CONFIG_9 = "config_9"
CONFIG_10 = "config_10"
CONFIG_11 = "config_11"
CONFIG_12 = "config_12"
CONFIG_13 = "config_13"

# (inference_mode, training_mode) for every stored configuration: every cell of
# the factorial ablation, including the two that make up the full method.
PAPER_CONFIGS = {
    CONFIG_BASE: (INF_NONE, TR_NONE),
    CONFIG_1: (INF_MARGIN, TR_NONE),
    CONFIG_2: (INF_MLHAT_A, TR_NONE),
    CONFIG_3: (INF_SQRTAM, TR_NONE),
    CONFIG_4: (INF_MMR, TR_NONE),
    CONFIG_5: (INF_NONE, TR_SELF_M),
    CONFIG_6: (INF_NONE, TR_DISAGREE_C),
    CONFIG_7: (INF_NONE, TR_TRAIN_C),
    CONFIG_8: (INF_NONE, TR_TRAIN_W),
    CONFIG_9: (INF_SQRTAM, TR_TRAIN_C),
    CONFIG_10: (INF_MMR, TR_TRAIN_C),
    CONFIG_11: (INF_SQRTAM, TR_TRAIN_W),
    CONFIG_12: (INF_MMR, TR_TRAIN_W),
    CONFIG_13: (INF_NONE, TR_SELF_A),
}

# Names used in the paper for the configurations that appear in its tables and
# figures. Each one names the mechanism the cell switches on, so a row of the
# table can be read as "what does this mechanism buy" rather than as a contest
# between implementations.
METHOD_NAMES = {
    CONFIG_BASE: "Uniform",        # no selection, no self-training
    CONFIG_1: "Margin",            # selection by prediction margin
    CONFIG_2: "RefereeAcc",        # selection by referee-estimated accuracy
    CONFIG_3: "Competence",        # selection by sqrt(A-hat * M)
    CONFIG_4: "MMR",               # selection by maximal marginal relevance
    CONFIG_5: "SelfTrain-M",       # self-training gated by margin
    CONFIG_6: "Disagree",          # learning by disagreement
    CONFIG_13: "SelfTrain-A",      # self-training gated by referee accuracy
    CONFIG_SLEADE: "SLEADE",       # external reference implementation
    CONFIG_10: "LENS-M",           # MMR + disagreement, consensus weighting
    CONFIG_12: "LENS",             # the full method
}
