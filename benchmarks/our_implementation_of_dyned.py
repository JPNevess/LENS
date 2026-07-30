"""Accuracy and diversity managed together, in the spirit of DynED.

Members are selected by maximal marginal relevance, which trades estimated
competence against pairwise diversity, so the selected subset is accurate and
complementary rather than merely confident. Unlabelled instances are discarded,
which isolates the selection mechanism from any semi-supervised effect.

Reference: Abadifard et al., DynED: Dynamic ensemble diversification in data
stream classification, CIKM 2023.
"""
from _runner import Method, main
from lens.config import CONFIG_4, INF_MMR, TR_NONE

METHOD = Method(
    key=CONFIG_4,
    name="DynED",
    reference="Abadifard et al., CIKM 2023",
    inference_mode=INF_MMR,
    training_mode=TR_NONE,
)

if __name__ == "__main__":
    main(METHOD)
