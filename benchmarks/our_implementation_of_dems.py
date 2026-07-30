"""Competence combined with margin, in the spirit of DEMS.

Members are scored by the geometric mean of the meta-learned accuracy and the
learner's own prediction margin, sqrt(A_hat * M). Selection is by relevance
only: there is no diversity term, so redundant members can be selected together.
Unlabelled instances are discarded.

Reference: Sun et al., Dynamic Ensemble Member Selection for Data Stream
Classification, CIKM 2025.
"""
from _runner import Method, main
from lens.config import CONFIG_3, INF_SQRTAM, TR_NONE

METHOD = Method(
    key=CONFIG_3,
    name="DEMS",
    reference="Sun et al., CIKM 2025",
    inference_mode=INF_SQRTAM,
    training_mode=TR_NONE,
)

if __name__ == "__main__":
    main(METHOD)
