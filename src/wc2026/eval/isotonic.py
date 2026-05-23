"""Per-outcome isotonic recalibration of 1X2 forecasts.

The Poisson-DC model is reasonably sharp but the reliability diagram on WC
hindcasts shows a stretching effect — predicted probabilities near 0 and 1
are systematically too extreme. Isotonic regression is the textbook
non-parametric recalibrator: it learns a monotone map ``p_raw -> p_calibrated``
per outcome from observed (p_raw, indicator) pairs.

We fit three independent monotone regressions — one for each of ``H``, ``D``,
``A`` — and then re-normalise the three corrected probabilities to sum to 1.
That re-normalisation is a small approximation (the joint calibration is not
guaranteed to remain proper), but in practice it gives a meaningful log-loss
reduction at almost no implementation cost (Niculescu-Mizil & Caruana 2005;
also used by 538's NFL ELO model).

We use scikit-learn's :class:`sklearn.isotonic.IsotonicRegression` with
``y_min=0``, ``y_max=1``, ``out_of_bounds="clip"``. We also add a small floor
(``EPS_PROB``) before re-normalisation so a calibrated probability of 0 cannot
kill log-loss for the held-out match.

The floor is intentionally generous (``0.01`` = 1%). A sub-1% machine-epsilon
floor (the previous ``1e-6``) made LOO log-loss EXPLODE on small samples like
WC 2022 (N=64): the isotonic step function would map some inputs to near-zero,
and a single "wrong-side" actual outcome contributed ``-log(1e-6) ≈ 13.8`` to
the mean log-loss — a +0.20 hit per match. 1% caps that per-match contribution
at ``-log(0.01) = 4.6``, which is the realistic "any team can beat any other
team about 1 in 100 times" floor for international football.

Leave-one-out evaluation: for an N-row dataset, fit on N-1 and apply to the
held-out row, repeated for every row. Used by the hindcast script so the
reported numbers do not leak observed outcomes into the calibrator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

EPS_PROB: float = 0.01


class IsotonicCalibrator:
    """Fitted per-outcome monotone calibrators for 1X2 forecasts.

    Use :meth:`fit` then :meth:`transform`; both take a DataFrame with the
    columns ``p_home``, ``p_draw``, ``p_away``, plus ``observed`` (H/D/A)
    for ``fit``.
    """

    def __init__(self) -> None:
        self.iso_h_: IsotonicRegression | None = None
        self.iso_d_: IsotonicRegression | None = None
        self.iso_a_: IsotonicRegression | None = None
        self.n_train_: int = 0

    @property
    def fitted(self) -> bool:
        return self.iso_h_ is not None

    def fit(self, predictions: pd.DataFrame) -> IsotonicCalibrator:
        required = {"p_home", "p_draw", "p_away", "observed"}
        missing = required - set(predictions.columns)
        if missing:
            raise ValueError(f"predictions missing columns: {sorted(missing)}")
        if predictions.empty:
            raise ValueError("predictions is empty")

        obs = predictions["observed"].to_numpy()
        self.iso_h_ = _fit_one(predictions["p_home"].to_numpy(), (obs == "H").astype(float))
        self.iso_d_ = _fit_one(predictions["p_draw"].to_numpy(), (obs == "D").astype(float))
        self.iso_a_ = _fit_one(predictions["p_away"].to_numpy(), (obs == "A").astype(float))
        self.n_train_ = len(predictions)
        return self

    def transform(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Apply the calibrators and re-normalise to a valid distribution."""
        if not self.fitted:
            raise RuntimeError("IsotonicCalibrator not fitted; call fit() first")
        required = {"p_home", "p_draw", "p_away"}
        missing = required - set(predictions.columns)
        if missing:
            raise ValueError(f"predictions missing columns: {sorted(missing)}")

        out = predictions.copy()
        ph = self.iso_h_.predict(out["p_home"].to_numpy())
        pd_ = self.iso_d_.predict(out["p_draw"].to_numpy())
        pa = self.iso_a_.predict(out["p_away"].to_numpy())
        stacked = np.stack([ph, pd_, pa], axis=1)
        stacked = np.clip(stacked, EPS_PROB, 1.0)
        stacked = stacked / stacked.sum(axis=1, keepdims=True)
        out["p_home"] = stacked[:, 0]
        out["p_draw"] = stacked[:, 1]
        out["p_away"] = stacked[:, 2]
        return out


def _fit_one(p_raw: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
    iso.fit(p_raw, y)
    return iso


def leave_one_out_recalibrate(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return ``predictions`` with each row's probs replaced by an LOO-calibrated copy.

    For each row ``i``, fit a calibrator on all other rows and apply it to row
    ``i``. The output preserves the input row order. This is the honest way to
    report calibrated log-loss on the same data used to fit the calibrator —
    no information from row ``i``'s observed outcome leaks into row ``i``'s
    calibrated probabilities.

    Complexity is ``O(N)`` calibrator fits, each ``O(N log N)``. For N≈64 (a
    World Cup) this is negligible.
    """
    required = {"p_home", "p_draw", "p_away", "observed"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    n = len(predictions)
    if n < 3:
        raise ValueError(f"need >= 3 rows for leave-one-out, got {n}")

    out = predictions.copy().reset_index(drop=True)
    new_h = np.empty(n)
    new_d = np.empty(n)
    new_a = np.empty(n)
    for i in range(n):
        train = out.drop(index=i)
        cal = IsotonicCalibrator().fit(train)
        applied = cal.transform(out.iloc[[i]])
        new_h[i] = float(applied["p_home"].iloc[0])
        new_d[i] = float(applied["p_draw"].iloc[0])
        new_a[i] = float(applied["p_away"].iloc[0])
    out["p_home"] = new_h
    out["p_draw"] = new_d
    out["p_away"] = new_a
    return out
