"""Tests for the Platt-scaling recalibrator."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wc2026.eval.platt import EPS_PROB, PlattCalibrator


def _synthetic_predictions(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """A toy dataset where the raw model is mildly overconfident on home wins.

    Construction: draw p_home ~ Uniform(0.1, 0.9), then sample the actual
    outcome from a slightly less-confident distribution. Platt should
    learn to pull the raw probabilities toward 0.5.
    """
    rng = np.random.default_rng(seed)
    p_home = rng.uniform(0.1, 0.9, size=n)
    # True probability is 0.6 * p_home + 0.2 (always between 0.26 and 0.74).
    true_p_home = 0.6 * p_home + 0.2
    u = rng.uniform(size=n)
    is_home = u < true_p_home
    is_draw = ~is_home & (rng.uniform(size=n) < 0.3)
    observed = np.where(is_home, "H", np.where(is_draw, "D", "A"))
    # Mock per-outcome raws — leave draws + aways simple.
    p_draw = rng.uniform(0.15, 0.35, size=n)
    p_away = 1.0 - p_home - p_draw
    # Re-normalise to a valid distribution.
    stacked = np.stack([p_home, p_draw, p_away], axis=1)
    stacked = stacked / stacked.sum(axis=1, keepdims=True)
    return pd.DataFrame(
        {
            "p_home": stacked[:, 0],
            "p_draw": stacked[:, 1],
            "p_away": stacked[:, 2],
            "observed": observed,
        }
    )


def test_fit_sets_six_coefficients_and_records_n_train() -> None:
    df = _synthetic_predictions(150)
    cal = PlattCalibrator().fit(df)
    assert cal.fitted is True
    assert cal.n_train_ == 150
    assert cal.slope_h_ is not None and cal.slope_d_ is not None and cal.slope_a_ is not None
    assert cal.intercept_h_ is not None
    assert cal.intercept_d_ is not None
    assert cal.intercept_a_ is not None


def test_transform_outputs_rows_that_sum_to_one() -> None:
    df = _synthetic_predictions(120, seed=1)
    cal = PlattCalibrator().fit(df)
    out = cal.transform(df.head(20))
    sums = (out["p_home"] + out["p_draw"] + out["p_away"]).to_numpy()
    np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-9)


def test_transform_applies_eps_floor() -> None:
    """A row whose raw probs are extreme should still respect the EPS floor."""
    df = pd.DataFrame(
        {
            "p_home": [0.999],
            "p_draw": [0.0005],
            "p_away": [0.0005],
            "observed": ["H"],
        }
    )
    df = pd.concat([df] * 50, ignore_index=True)
    cal = PlattCalibrator().fit(df)
    out = cal.transform(df.head(1))
    for col in ("p_home", "p_draw", "p_away"):
        # No probability should fall below the floor / (1 + 2 * floor) (== floor
        # after re-normalisation).
        assert float(out.iloc[0][col]) >= EPS_PROB / (1.0 + 2 * EPS_PROB) - 1e-9


def test_transform_requires_fit() -> None:
    df = _synthetic_predictions(10)
    cal = PlattCalibrator()
    with pytest.raises(RuntimeError, match="not fitted"):
        cal.transform(df)


def test_fit_rejects_missing_columns() -> None:
    bad = pd.DataFrame({"p_home": [0.5], "p_draw": [0.3], "observed": ["H"]})
    cal = PlattCalibrator()
    with pytest.raises(ValueError, match="missing columns"):
        cal.fit(bad)


def test_fit_rejects_empty_predictions() -> None:
    empty = pd.DataFrame(
        columns=["p_home", "p_draw", "p_away", "observed"]
    )
    cal = PlattCalibrator()
    with pytest.raises(ValueError, match="empty"):
        cal.fit(empty)


def test_single_class_outcome_falls_back_to_base_rate_intercept() -> None:
    """If an outcome class has zero positives, the per-outcome fit should
    still produce sensible coefficients (zero slope + logit-of-base-rate
    intercept) — sklearn's LogisticRegression refuses to fit otherwise."""
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame(
        {
            "p_home": rng.uniform(0.4, 0.6, size=n),
            "p_draw": rng.uniform(0.2, 0.3, size=n),
            "p_away": rng.uniform(0.2, 0.3, size=n),
            "observed": ["H"] * n,  # zero draws + zero aways
        }
    )
    cal = PlattCalibrator().fit(df)
    # Draw + away both have zero positives → slope = 0.
    assert cal.slope_d_ == 0.0
    assert cal.slope_a_ == 0.0
    # The intercept is the logit of the floored base rate (EPS_PROB ≈ 0.01).
    expected = math.log(EPS_PROB / (1.0 - EPS_PROB))
    assert math.isclose(cal.intercept_d_, expected, rel_tol=1e-6)
    assert math.isclose(cal.intercept_a_, expected, rel_tol=1e-6)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    df = _synthetic_predictions(80, seed=42)
    cal = PlattCalibrator().fit(df)
    artifact = tmp_path / "platt.npz"
    cal.save(artifact)

    cal2 = PlattCalibrator.load(artifact)
    assert cal2.n_train_ == cal.n_train_
    for attr in (
        "slope_h_",
        "intercept_h_",
        "slope_d_",
        "intercept_d_",
        "slope_a_",
        "intercept_a_",
    ):
        assert math.isclose(getattr(cal2, attr), getattr(cal, attr), rel_tol=1e-12)
    # Same transformed output.
    a = cal.transform(df.head(5))
    b = cal2.transform(df.head(5))
    np.testing.assert_allclose(a["p_home"].to_numpy(), b["p_home"].to_numpy())


def test_save_unfitted_raises(tmp_path: Path) -> None:
    cal = PlattCalibrator()
    with pytest.raises(RuntimeError, match="nothing to save"):
        cal.save(tmp_path / "platt.npz")


# --- WC2026_USE_PLATT env-flag wiring ----------------------------------------


def test_get_platt_returns_none_when_env_flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from wc2026.api.routes import predictions as pred_mod

    monkeypatch.delenv(pred_mod.PLATT_ENV_FLAG, raising=False)
    pred_mod.reset_platt_cache()
    assert pred_mod._get_platt() is None


def test_get_platt_returns_none_when_artifact_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from wc2026.api.routes import predictions as pred_mod

    monkeypatch.setenv(pred_mod.PLATT_ENV_FLAG, "1")
    monkeypatch.setattr(pred_mod, "PLATT_ARTIFACT_PATH", tmp_path / "absent.npz")
    pred_mod.reset_platt_cache()
    assert pred_mod._get_platt() is None


def test_maybe_apply_platt_passthrough_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wiring must be a true pass-through when the flag is off — Platt
    artifacts may not exist, but predictions must keep working."""
    from wc2026.api.routes import predictions as pred_mod
    from wc2026.api.schemas import OutcomeProbabilities

    monkeypatch.delenv(pred_mod.PLATT_ENV_FLAG, raising=False)
    pred_mod.reset_platt_cache()
    out = OutcomeProbabilities(home_win=0.5, draw=0.3, away_win=0.2)
    got = pred_mod._maybe_apply_platt(out)
    assert got is out  # exact same instance returned — proves no-op fast path


def test_maybe_apply_platt_calibrates_when_artifact_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from wc2026.api.routes import predictions as pred_mod
    from wc2026.api.schemas import OutcomeProbabilities

    df = _synthetic_predictions(120, seed=7)
    cal = PlattCalibrator().fit(df)
    artifact = tmp_path / "platt.npz"
    cal.save(artifact)

    monkeypatch.setenv(pred_mod.PLATT_ENV_FLAG, "1")
    monkeypatch.setattr(pred_mod, "PLATT_ARTIFACT_PATH", artifact)
    pred_mod.reset_platt_cache()

    raw = OutcomeProbabilities(home_win=0.6, draw=0.2, away_win=0.2)
    got = pred_mod._maybe_apply_platt(raw)
    # Output must be a valid 1X2 distribution.
    s = got.home_win + got.draw + got.away_win
    assert math.isclose(s, 1.0, rel_tol=1e-6)
    # And it must NOT be the literal input (calibrator did something).
    assert not (
        math.isclose(got.home_win, raw.home_win)
        and math.isclose(got.draw, raw.draw)
        and math.isclose(got.away_win, raw.away_win)
    )
