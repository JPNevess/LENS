"""Meta-learned self-training, in the spirit of learning-to-self-train.

Both the decision to accept a pseudo-label and its weight come from the
meta-learned accuracy A_hat alone, without the prediction margin and without
ensemble disagreement; the meta-learner is an incremental decision tree rather
than a gradient-based one. The vote is a uniform majority, which isolates the
self-training signal.

Reference: Li et al., Learning to self-train for semi-supervised few-shot
classification, NeurIPS 2019.
"""
from _runner import Method, main
from lens.config import CONFIG_13, INF_NONE, TR_SELF_A

METHOD = Method(
    key=CONFIG_13,
    name="LST*",
    reference="Li et al., NeurIPS 2019",
    inference_mode=INF_NONE,
    training_mode=TR_SELF_A,
)

if __name__ == "__main__":
    main(METHOD)
