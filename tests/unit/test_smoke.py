"""Smoke tests: the package imports, the toolchain is alive."""

from __future__ import annotations

import sys


def test_python_version() -> None:
    assert sys.version_info >= (3, 12), f"need Python >= 3.12, have {sys.version_info}"


def test_package_imports() -> None:
    import wc2026

    assert wc2026.__version__ == "0.1.0"


def test_core_scientific_stack_imports() -> None:
    import numpy as np
    import pandas as pd
    import scipy
    import statsmodels.api as sm

    assert np.__version__
    assert pd.__version__
    assert scipy.__version__
    assert sm is not None
