"""Selection and self-training modes, and the identifiers used in results files.

The ensemble is parametrised along two orthogonal axes:

* ``inference_mode`` -- how member votes are weighted and selected at prediction
  time (dynamic ensemble selection).
* ``training_mode`` -- how unlabelled instances are turned into pseudo-labels and
  weighted during training (semi-supervised learning).

Each benchmark in ``benchmarks/`` is one point in this space. The ``config_*``
identifiers below are the keys under which every run is stored in ``results/``,
so they are kept stable; ``METHOD_NAMES`` maps them to the names used in the
paper.
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

# (inference_mode, training_mode) for every stored configuration. The first nine
# reproduce a published mechanism; the last four are the cells of the factorial
# ablation that are not baselines.
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
# figures.
METHOD_NAMES = {
    CONFIG_BASE: "ARF",
    CONFIG_1: "DyAbst",
    CONFIG_2: "Meta-BR",
    CONFIG_3: "DEMS",
    CONFIG_4: "DynED",
    CONFIG_5: "Self-train",
    CONFIG_6: "SCo-For",
    CONFIG_13: "LST*",
    CONFIG_SLEADE: "SLEADE",
    CONFIG_10: "LENS-M",
    CONFIG_12: "LENS",
}
