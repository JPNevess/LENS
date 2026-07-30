"""LENS: a semi-supervised streaming ensemble over a multi-target competence map."""
from .config import PAPER_CONFIGS, METHOD_NAMES
from .ensemble import LENS
from .evaluation import run_experiment

__all__ = ["LENS", "run_experiment", "PAPER_CONFIGS", "METHOD_NAMES"]
