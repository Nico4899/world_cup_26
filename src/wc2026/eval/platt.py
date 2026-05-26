"""Per-outcome Platt scaling for 1X2 forecasts.

Mirrors :class:`wc2026.eval.isotonic.IsotonicCalibrator`'s shape so the
two recalibrators are interchangeable behind the same interface. Platt
scaling fits a single logistic regression per outcome (``H``, ``D``,
``A``) on the ``(p_raw, indicator)`` pairs, then re-normalises the three
calibrated probabilities to sum to 1.

Why a separate calibrator
-------------------------
The existing isotonic calibrator is fragile on small samples — LOO on
WC 2022 (N=64) degrades log-loss by +0.077, the step-function nature
gets unlucky on a single tail bin. Platt has only 2 parameters per
outcome (intercept + slope) so it's a much smaller capacity model;
empirical sports-AI reports (Sanjay 2024 EPL example, Niculescu-Mizil
& Caruana 2005) show it tends to improve Brier + ECE by 0.01-0.03 on
multi-tournament corpora.

Status: shipped, **OFF by default**. The ``WC2026_USE_PLATT`` env flag
in :mod:`wc2026.api.routes.predictions` gates whether prediction
responses are passed through this calibrator. Roll out only after the
multi-tournament corpus is large enough that the gate holds (≥ 0.002
log-loss improvement on the WC 2018 + WC 2022 holdout).

Serialization
-------------
``save`` / ``load`` write a NumPy ``.npz`` with six scalars (slope +
intercept per outcome) + ``n_train``. We intentionally do **not** pickle
the scikit-learn estimator — the file would tie us to a specific
sklearn version, and the underlying logistic regression has all the
state we need in two parameters.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

EPS_PROB: float = 0.01
"""Floor applied to each calibrated probability before re-normalisation.

Same value the isotonic calibrator uses — caps any single mis-aligned
forecast's contribution to log-loss at ``-log(0.01) ≈ 4.6`` rather than
letting a near-zero spike dominate the mean.
"""


class PlattCalibrator:
    """Fitted per-outcome logistic recalibrators for 1X2 forecasts.

    Coefficients are stored as plain floats so the calibrator can be
    round-tripped through ``.npz`` without pickling a scikit estimator.
    """

    def __init__(self) -> None:
        self.slope_h_: float | None = None
        self.intercept_h_: float | None = None
        self.slope_d_: float | None = None
        self.intercept_d_: float | None = None
        self.slope_a_: float | None = None
        self.intercept_a_: float | None = None
        self.n_train_: int = 0

    @property
    def fitted(self) -> bool:
        return self.slope_h_ is not None

    def fit(self, predictions: pd.DataFrame) -> PlattCalibrator:
        required = {"p_home", "p_draw", "p_away", "observed"}
        missing = required - set(predictions.columns)
        if missing:
            raise ValueError(f"predictions missing columns: {sorted(missing)}")
        if predictions.empty:
            raise ValueError("predictions is empty")

        obs = predictions["observed"].to_numpy()
        ph_slope, ph_intercept = _fit_one(
            predictions["p_home"].to_numpy(), (obs == "H").astype(int)
        )
        pd_slope, pd_intercept = _fit_one(
            predictions["p_draw"].to_numpy(), (obs == "D").astype(int)
        )
        pa_slope, pa_intercept = _fit_one(
            predictions["p_away"].to_numpy(), (obs == "A").astype(int)
        )
        self.slope_h_ = ph_slope
        self.intercept_h_ = ph_intercept
        self.slope_d_ = pd_slope
        self.intercept_d_ = pd_intercept
        self.slope_a_ = pa_slope
        self.intercept_a_ = pa_intercept
        self.n_train_ = len(predictions)
        return self

    def transform(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Apply the calibrators and re-normalise to a valid distribution."""
        if not self.fitted:
            raise RuntimeError("PlattCalibrator not fitted; call fit() first")
        required = {"p_home", "p_draw", "p_away"}
        missing = required - set(predictions.columns)
        if missing:
            raise ValueError(f"predictions missing columns: {sorted(missing)}")

        out = predictions.copy()
        ph = _sigmoid(self.slope_h_, self.intercept_h_, out["p_home"].to_numpy())
        pd_ = _sigmoid(self.slope_d_, self.intercept_d_, out["p_draw"].to_numpy())
        pa = _sigmoid(self.slope_a_, self.intercept_a_, out["p_away"].to_numpy())
        stacked = np.stack([ph, pd_, pa], axis=1)
        stacked = np.clip(stacked, EPS_PROB, 1.0)
        stacked = stacked / stacked.sum(axis=1, keepdims=True)
        out["p_home"] = stacked[:, 0]
        out["p_draw"] = stacked[:, 1]
        out["p_away"] = stacked[:, 2]
        return out

    # ------------------------------------------------------------------ I/O

    def save(self, path: Path) -> None:
        if not self.fitted:
            raise RuntimeError("PlattCalibrator not fitted; nothing to save")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            slope_h=np.float64(self.slope_h_),
            intercept_h=np.float64(self.intercept_h_),
            slope_d=np.float64(self.slope_d_),
            intercept_d=np.float64(self.intercept_d_),
            slope_a=np.float64(self.slope_a_),
            intercept_a=np.float64(self.intercept_a_),
            n_train=np.int64(self.n_train_),
        )

    @classmethod
    def load(cls, path: Path) -> PlattCalibrator:
        with np.load(path) as npz:
            cal = cls()
            cal.slope_h_ = float(npz["slope_h"])
            cal.intercept_h_ = float(npz["intercept_h"])
            cal.slope_d_ = float(npz["slope_d"])
            cal.intercept_d_ = float(npz["intercept_d"])
            cal.slope_a_ = float(npz["slope_a"])
            cal.intercept_a_ = float(npz["intercept_a"])
            cal.n_train_ = int(npz["n_train"])
        return cal


def _fit_one(p_raw: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Single-feature logistic regression on ``(p_raw, y)``.

    Falls back to a degenerate (zero-slope, intercept = logit of the
    base rate) coefficient pair when ``y`` is single-valued — sklearn
    refuses to fit a logistic in that case, but the calibrator must
    still be transformable downstream.
    """
    unique = np.unique(y)
    if len(unique) < 2:
        base = float(np.mean(y))
        # logit(p) with floor / ceil to avoid log(0).
        clipped = min(max(base, EPS_PROB), 1.0 - EPS_PROB)
        return 0.0, float(math.log(clipped / (1.0 - clipped)))
    model = LogisticRegression(solver="lbfgs", C=1.0, max_iter=200)
    model.fit(p_raw.reshape(-1, 1), y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def _sigmoid(slope: float | None, intercept: float | None, x: np.ndarray) -> np.ndarray:
    """Logistic sigmoid ``1 / (1 + exp(-(slope * x + intercept)))``.

    Pre-condition: the calibrator is fitted, so ``slope`` and ``intercept``
    are never None when this is reached. Caller (:meth:`transform`)
    enforces that via the ``fitted`` check.
    """
    assert slope is not None and intercept is not None
    z = slope * x + intercept
    # np.clip on z avoids overflow on extreme inputs (matters when the
    # corpus pushes slope above ~10).
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


__all__ = ["PlattCalibrator", "EPS_PROB"]
