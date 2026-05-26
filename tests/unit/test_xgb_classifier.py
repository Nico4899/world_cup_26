"""Unit tests for the XGB H/D/A match classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wc2026.models.xgb_classifier import (
    CLASS_AWAY,
    CLASS_DRAW,
    CLASS_HOME,
    DEFAULT_FEATURE_COLUMNS,
    XgbMatchModel,
    label_from_score,
    labels_for_matches,
)


def _synthetic_corpus(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """A 3-class corpus where elo_diff is the dominant signal."""
    rng = np.random.default_rng(seed)
    elo_diff = rng.normal(0.0, 120.0, size=n)
    fifa_rank_diff = -elo_diff / 30.0 + rng.normal(0.0, 5.0, size=n)
    xg_form_diff = elo_diff / 200.0 + rng.normal(0.0, 0.3, size=n)
    rest_days_diff = rng.integers(-3, 4, size=n)
    squad_age_diff = rng.normal(0.0, 1.5, size=n)
    is_neutral = rng.integers(0, 2, size=n)
    is_host_home = rng.integers(0, 2, size=n)
    is_host_away = rng.integers(0, 2, size=n)
    poisson_exp_home = 1.4 + 0.004 * elo_diff + rng.normal(0, 0.2, size=n)
    poisson_exp_away = 1.2 - 0.004 * elo_diff + rng.normal(0, 0.2, size=n)
    poisson_p_home = np.clip(0.4 + 0.0015 * elo_diff + rng.normal(0, 0.05, size=n), 0.05, 0.9)
    poisson_p_draw = np.clip(0.27 + rng.normal(0, 0.04, size=n), 0.05, 0.5)
    poisson_p_away = np.clip(1 - poisson_p_home - poisson_p_draw, 0.05, 0.9)

    # True label: heavily driven by elo_diff with bounded noise.
    logits_h = 0.018 * elo_diff
    logits_a = -0.018 * elo_diff
    rng_label = rng.normal(0, 1.0, size=n)
    y = np.where(
        logits_h + rng_label > 1.0,
        CLASS_HOME,
        np.where(logits_a + rng_label > 1.0, CLASS_AWAY, CLASS_DRAW),
    ).astype(int)

    # Venue altitude + travel-km diff — included as low-signal columns so
    # XGB tests stay forward-compatible with the W2.1 + W2.2
    # DEFAULT_FEATURE_COLUMNS expansion.
    venue_altitude_m = rng.uniform(0.0, 2500.0, size=n)
    travel_km_diff = rng.normal(0.0, 1500.0, size=n)
    X = pd.DataFrame(
        {
            "elo_diff": elo_diff,
            "fifa_rank_diff": fifa_rank_diff,
            "xg_form_diff": xg_form_diff,
            "rest_days_diff": rest_days_diff,
            "squad_age_diff": squad_age_diff,
            "is_neutral": is_neutral,
            "is_host_home": is_host_home,
            "is_host_away": is_host_away,
            "poisson_exp_home_goals": poisson_exp_home,
            "poisson_exp_away_goals": poisson_exp_away,
            "poisson_p_home": poisson_p_home,
            "poisson_p_draw": poisson_p_draw,
            "poisson_p_away": poisson_p_away,
            "venue_altitude_m": venue_altitude_m,
            "travel_km_diff": travel_km_diff,
        }
    )
    return X, y


def test_label_from_score_home_draw_away() -> None:
    assert label_from_score(2, 1) == CLASS_HOME
    assert label_from_score(0, 0) == CLASS_DRAW
    assert label_from_score(1, 3) == CLASS_AWAY


def test_labels_for_matches_vectorised() -> None:
    matches = pd.DataFrame({"home_score": [3, 1, 1, 0], "away_score": [0, 2, 1, 0]})
    labels = labels_for_matches(matches)
    assert labels.tolist() == [CLASS_HOME, CLASS_AWAY, CLASS_DRAW, CLASS_DRAW]


def test_labels_for_matches_rejects_nan_score() -> None:
    matches = pd.DataFrame({"home_score": [3, None], "away_score": [0, 0]})
    with pytest.raises(ValueError, match="NaN"):
        labels_for_matches(matches)


def test_fit_returns_model_with_canonical_feature_order() -> None:
    X, y = _synthetic_corpus()
    model = XgbMatchModel.fit(X, y)
    assert model.feature_names == DEFAULT_FEATURE_COLUMNS


def test_predict_proba_sums_to_one_per_row() -> None:
    X, y = _synthetic_corpus()
    model = XgbMatchModel.fit(X, y)
    probs = model.predict_proba(X.head(50))
    assert probs.shape == (50, 3)
    sums = probs.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-6)


def test_predict_proba_responds_to_elo_diff() -> None:
    X, y = _synthetic_corpus(n=1500, seed=1)
    model = XgbMatchModel.fit(X, y)
    pos = X.copy()
    pos["elo_diff"] = 200.0
    neg = X.copy()
    neg["elo_diff"] = -200.0
    pos_prob = model.predict_proba(pos)
    neg_prob = model.predict_proba(neg)
    # Strong elo advantage → higher home-win probability than strong disadvantage.
    assert pos_prob[:, CLASS_HOME].mean() > neg_prob[:, CLASS_HOME].mean() + 0.1


def test_predict_proba_handles_missing_columns_with_nan() -> None:
    X, y = _synthetic_corpus(n=400, seed=2)
    model = XgbMatchModel.fit(X, y)
    df = X.head(8).drop(columns=["rest_days_diff", "squad_age_diff"])
    probs = model.predict_proba(df)
    assert probs.shape == (8, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    X, y = _synthetic_corpus(seed=3)
    model = XgbMatchModel.fit(X, y)
    model_path = tmp_path / "model.json"
    meta_path = tmp_path / "meta.json"
    model.save(model_path, meta_path)
    reloaded = XgbMatchModel.load(model_path, meta_path)
    p1 = model.predict_proba(X.head(20))
    p2 = reloaded.predict_proba(X.head(20))
    assert np.allclose(p1, p2)
    assert reloaded.feature_names == model.feature_names
    assert reloaded.version == model.version


def test_fit_uses_sample_weights() -> None:
    """Sample weights should change the learned model; we just verify the fit
    accepts the kwarg and produces a different model than the unweighted fit."""
    X, y = _synthetic_corpus(seed=4)
    w = np.where(y == CLASS_DRAW, 5.0, 1.0)  # over-weight draws
    weighted = XgbMatchModel.fit(X, y, sample_weight=w)
    unweighted = XgbMatchModel.fit(X, y)
    p_weighted = weighted.predict_proba(X.head(50))
    p_unweighted = unweighted.predict_proba(X.head(50))
    # The weighted model should assign higher draw probability on average.
    assert p_weighted[:, CLASS_DRAW].mean() > p_unweighted[:, CLASS_DRAW].mean()


def test_fit_raises_when_required_column_missing() -> None:
    X, y = _synthetic_corpus()
    bad = X.drop(columns=["xg_form_diff"])
    with pytest.raises(ValueError, match="missing"):
        XgbMatchModel.fit(bad, y)
