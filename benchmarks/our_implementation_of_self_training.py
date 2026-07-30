"""Confidence-based self-training.

For an unlabelled instance, each member trains on its own prediction whenever
its margin clears a threshold, with the update weighted by that margin and by
the label density. The vote is a uniform majority, so any gain comes from the
self-training signal alone.

Reference: Le Nguyen, Gomes and Bifet, Semi-supervised learning over streaming
data using MOA, IEEE Big Data 2019.
"""
from _runner import Method, main
from lens.config import CONFIG_5, INF_NONE, TR_SELF_M

METHOD = Method(
    key=CONFIG_5,
    name="Self-train",
    reference="Le Nguyen, Gomes and Bifet, IEEE Big Data 2019",
    inference_mode=INF_NONE,
    training_mode=TR_SELF_M,
)

if __name__ == "__main__":
    main(METHOD)
