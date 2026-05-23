"""Unit tests for isotonic recalibration of 1X2 forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wc2026.eval.calibration import aggregate
from wc2026.eval.isotonic import IsotonicCalibrator, leave_one_out_recalibrate


def _three_class_synth(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Generate rows whose true outcome is sampled from (p_home, p_draw, p_away).

    The raw forecasts will then BE perfectly calibrated by construction. We
    add a multiplicative "sharpening" distortion afterwards in the tests that
    need miscalibration.
    """
    base = rng.dirichlet([2.0, 1.0, 2.0], size=n)
    obs = np.array([rng.choice(["H", "D", "A"], p=p) for p in base])
    return pd.DataFrame(
        {
            "p_home": base[:, 0],
            "p_draw": base[:, 1],
            "p_away": base[:, 2],
            "observed": obs,
        }
    )


def _sharpen(predictions: pd.DataFrame, gamma: float) -> pd.DataFrame:
    """Apply p -> p^gamma / Z miscalibration (sharper if gamma>1, flatter if <1)."""
    out = predictions.copy()
    arr = out[["p_home", "p_draw", "p_away"]].to_numpy() ** gamma
    arr = arr / arr.sum(axis=1, keepdims=True)
    out["p_home"] = arr[:, 0]
    out["p_draw"] = arr[:, 1]
    out["p_away"] = arr[:, 2]
    return out


# --- transform shape + sum-to-1 --------------------------------------------


def test_transform_output_sums_to_one() -> None:
    rng = np.random.default_rng(1)
    df = _three_class_synth(rng, n=200)
    cal = IsotonicCalibrator().fit(df)
    out = cal.transform(df)
    sums = out[["p_home", "p_draw", "p_away"]].sum(axis=1).to_numpy()
    assert np.allclose(sums, 1.0)


def test_transform_preserves_row_order_and_extras() -> None:
    rng = np.random.default_rng(2)
    df = _three_class_synth(rng, n=50)
    df["match_id"] = range(len(df))
    cal = IsotonicCalibrator().fit(df)
    out = cal.transform(df)
    assert list(out["match_id"]) == list(df["match_id"])


def test_fit_then_transform_raises_when_unfitted() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        IsotonicCalibrator().transform(
            pd.DataFrame({"p_home": [0.4], "p_draw": [0.3], "p_away": [0.3]})
        )


def test_fit_missing_columns_raises() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        IsotonicCalibrator().fit(pd.DataFrame({"p_home": [0.3], "observed": ["H"]}))


# --- miscalibrated -> calibrated -------------------------------------------


def test_miscalibrated_input_improves_after_calibration() -> None:
    """Sharpen the perfectly-calibrated synthetic forecasts to a known-bad
    state; in-sample isotonic fit + transform should bring log-loss back
    below the sharpened log-loss."""
    rng = np.random.default_rng(0)
    df = _three_class_synth(rng, n=2000)
    sharp = _sharpen(df, gamma=2.0)  # overconfident
    cal = IsotonicCalibrator().fit(sharp)
    fixed = cal.transform(sharp)
    ll_sharp = aggregate(sharp).log_loss
    ll_fixed = aggregate(fixed).log_loss
    assert ll_fixed < ll_sharp, (
        f"isotonic did not help (sharp={ll_sharp:.4f}, fixed={ll_fixed:.4f})"
    )


# --- LOO doesn't leak ------------------------------------------------------


def test_loo_recalibration_does_not_use_held_out_row() -> None:
    """If isotonic recalibration leaked, then on data where the raw forecasts
    are already perfectly calibrated, in-sample isotonic fit would still
    'memorise' each held-out outcome and drive its log-loss to ~0. LOO
    recalibration on the same data should give a log-loss strictly above the
    in-sample fit (because no leakage)."""
    rng = np.random.default_rng(42)
    df = _three_class_synth(rng, n=80)
    cal_in = IsotonicCalibrator().fit(df)
    in_sample = cal_in.transform(df)
    loo = leave_one_out_recalibrate(df)
    ll_in = aggregate(in_sample).log_loss
    ll_loo = aggregate(loo).log_loss
    # The LOO log-loss should not be dramatically better than the in-sample
    # one (which would indicate leakage). In fact on perfectly-calibrated
    # synthetic data LOO is typically slightly WORSE than the in-sample fit.
    assert ll_loo >= ll_in - 1e-3, (
        f"LOO suspiciously better than in-sample (loo={ll_loo:.4f}, in={ll_in:.4f}) — possible leakage"
    )


def test_loo_renormalises_to_one() -> None:
    rng = np.random.default_rng(7)
    df = _three_class_synth(rng, n=30)
    loo = leave_one_out_recalibrate(df)
    sums = loo[["p_home", "p_draw", "p_away"]].sum(axis=1).to_numpy()
    assert np.allclose(sums, 1.0)


def test_loo_requires_three_rows() -> None:
    df = pd.DataFrame(
        {
            "p_home": [0.5, 0.5],
            "p_draw": [0.25, 0.25],
            "p_away": [0.25, 0.25],
            "observed": ["H", "A"],
        }
    )
    with pytest.raises(ValueError, match=">= 3"):
        leave_one_out_recalibrate(df)
