"""LENS: the full method.

Inference selects members by maximal marginal relevance, trading the relevance
sqrt(A_hat * M) against meta-learned pairwise diversity. Self-training admits a
pseudo-label when the rest of the ensemble is more confident than the member
itself and weights it by sqrt(A_bar * c). Drift detection monitors the same
meta-learned signals and, on a change, shifts lambda towards diversity while
weak members are replaced from the background pool.
"""
from _runner import Method, main
from lens.config import CONFIG_12, INF_MMR, TR_TRAIN_W

METHOD = Method(
    key=CONFIG_12,
    name="LENS",
    reference="this work",
    inference_mode=INF_MMR,
    training_mode=TR_TRAIN_W,
)

if __name__ == "__main__":
    main(METHOD)
