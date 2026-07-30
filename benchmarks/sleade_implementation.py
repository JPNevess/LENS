"""SLEADE baseline.

Runs the reference SLEADE implementation shipped with CapyMOA under the same
prequential semi-supervised protocol as every other method here, so this column
of the comparison uses the authors' own code rather than a reimplementation. It
does not go through the ensemble in ``lens``; see ``lens/baselines/sleade.py``.

Reference: Gomes et al., SLEADE: Disagreement-Based Semi-Supervised Learning for
Sparsely Labeled Evolving Data Streams, IEEE TKDE, 2025.
"""
from _runner import Method, main
from lens.config import CONFIG_SLEADE

METHOD = Method(
    key=CONFIG_SLEADE,
    name="SLEADE",
    reference="Gomes et al., IEEE TKDE, 2025",
    inference_mode=None,
    training_mode=None,
)

if __name__ == "__main__":
    main(METHOD)
