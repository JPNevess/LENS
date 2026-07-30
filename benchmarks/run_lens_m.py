"""LENS-M: the full selection mechanism with the relevance-based self-training
variant.

Inference selects members by maximal marginal relevance over sqrt(A_hat * M).
Self-training then reuses that same relevance signal to admit and weight
pseudo-labels, instead of the co-training-inspired disagreement signal used by
the full method. Comparing this against ``run_lens.py`` isolates the choice of
self-training signal while holding selection fixed.
"""
from _runner import Method, main
from lens.config import CONFIG_10, INF_MMR, TR_TRAIN_C

METHOD = Method(
    key=CONFIG_10,
    name="LENS-M",
    reference="this work",
    inference_mode=INF_MMR,
    training_mode=TR_TRAIN_C,
)

if __name__ == "__main__":
    main(METHOD)
