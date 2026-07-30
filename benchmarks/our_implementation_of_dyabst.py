"""Confidence-based dynamic weighting, in the spirit of DyAbst.

Members are weighted by their own prediction margin M, so a learner that is
unsure about the current instance contributes less to the vote. Unlabelled
instances are discarded.

Reference: Krawczyk and Cano, Online ensemble learning with abstaining
classifiers for drifting and noisy data streams, Applied Soft Computing 68, 2018.
"""
from _runner import Method, main
from lens.config import CONFIG_1, INF_MARGIN, TR_NONE

METHOD = Method(
    key=CONFIG_1,
    name="DyAbst",
    reference="Krawczyk and Cano, Applied Soft Computing 68, 2018",
    inference_mode=INF_MARGIN,
    training_mode=TR_NONE,
)

if __name__ == "__main__":
    main(METHOD)
