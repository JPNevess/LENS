"""Learn-by-disagreement, in the spirit of SCo-For.

A member is updated with the pseudo-label of the rest of the ensemble when the
others agree on it more confidently than the member believes its own prediction,
with the update weighted by that consensus confidence. This is the pure
disagreement signal, with no meta-learned component anywhere in the decision.
The vote is a uniform majority.

Reference: Wang and Li, Improving semi-supervised co-forest algorithm in
evolving data streams, Applied Intelligence 48, 2018.
"""
from _runner import Method, main
from lens.config import CONFIG_6, INF_NONE, TR_DISAGREE_C

METHOD = Method(
    key=CONFIG_6,
    name="SCo-For",
    reference="Wang and Li, Applied Intelligence 48, 2018",
    inference_mode=INF_NONE,
    training_mode=TR_DISAGREE_C,
)

if __name__ == "__main__":
    main(METHOD)
