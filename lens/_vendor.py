"""Make the vendored third-party trees importable.

``third_party/mlhat`` is the MLHAT reference implementation (Esteban et al.,
2024). Its modules import each other with absolute names, so the directory has
to be on ``sys.path``; it is left otherwise untouched.
"""
import os
import sys

_VENDOR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "third_party", "mlhat")


def ensure_vendor_path():
    if _VENDOR not in sys.path:
        sys.path.insert(0, _VENDOR)
    return _VENDOR
