"""ARF-style supervised baseline.

An ensemble of Hoeffding Adaptive Trees trained with random patches, that is
feature subsampling combined with Poisson bagging, aggregated by a uniform
majority vote. Unlabelled instances are discarded.

This is the reference point of the comparison rather than a port of the original
implementation: it shares the base learners, the bagging and the evaluation
protocol with every other method here, and differs only in having neither
dynamic selection nor self-training.

Reference: Gomes et al., Adaptive random forests for evolving data stream
classification, Machine Learning 106, 2017.
"""
from _runner import Method, main
from lens.config import CONFIG_BASE, INF_NONE, TR_NONE

METHOD = Method(
    key=CONFIG_BASE,
    name="ARF",
    reference="Gomes et al., Machine Learning 106, 2017",
    inference_mode=INF_NONE,
    training_mode=TR_NONE,
)

if __name__ == "__main__":
    main(METHOD)
