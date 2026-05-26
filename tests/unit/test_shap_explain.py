"""Unit tests for the SHAP wrapper around the XGB classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wc2026.models.shap_explain import (
    CLASS_NAMES,
    FeatureContribution,
    XgbExplainer,
)
from wc2026.models.xgb_classifier import (
    CLASS_AWAY,
    CLASS_DRAW,
    CLASS_HOME,
    DEFAULT_FEATURE_COLUMNS,
    XgbMatchModel,
)


def _synthetic_corpus(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Same shape as in test_xgb_classifier, dialled down to keep tests fast."""
    rng = np.random.default_rng(seed)
    elo_diff = rng.normal(0.0, 120.0, size=n)
    fifa_rank_diff = -elo_diff / 30.0
    xg_form_diff = elo_diff / 200.0
    poisson_p_home = np.clip(0.4 + 0.0015 * elo_diff, 0.05, 0.9)
    poisson_p_draw = np.full(n, 0.27)
    poisson_p_away = 1 - poisson_p_home - poisson_p_draw
    y = np.where(
        elo_diff > 50, CLASS_HOME, np.where(elo_diff < -50, CLASS_AWAY, CLASS_DRAW)
    ).astype(int)
    X = pd.DataFrame(
        {
            "elo_diff": elo_diff,
            "fifa_rank_diff": fifa_rank_diff,
            "xg_form_diff": xg_form_diff,
            "rest_days_diff": np.zeros(n),
            "squad_age_diff": np.zeros(n),
            "is_neutral": np.zeros(n, dtype=int),
            "is_host_home": np.zeros(n, dtype=int),
            "is_host_away": np.zeros(n, dtype=int),
            "poisson_exp_home_goals": 1.4 + 0.004 * elo_diff,
            "poisson_exp_away_goals": 1.2 - 0.004 * elo_diff,
            "poisson_p_home": poisson_p_home,
            "poisson_p_draw": poisson_p_draw,
            "poisson_p_away": poisson_p_away,
            # W2.1 / W2.2 — included so the synthetic dataset matches the
            # canonical DEFAULT_FEATURE_COLUMNS list. Zero variance is
            # intentional; we don't want SHAP attributing signal to them
            # in tests.
            "venue_altitude_m": np.zeros(n),
            "travel_km_diff": np.zeros(n),
        }
    )
    return X, y


def test_class_names_cover_three_outcomes() -> None:
    assert CLASS_NAMES == {CLASS_HOME: "home_win", CLASS_DRAW: "draw", CLASS_AWAY: "away_win"}


def test_explainer_attaches_to_a_fitted_model() -> None:
    X, y = _synthetic_corpus()
    model = XgbMatchModel.fit(X, y)
    explainer = XgbExplainer.from_model(model)
    assert explainer.model is model
    assert explainer.explainer is not None


def test_explain_row_returns_contribution_per_feature() -> None:
    X, y = _synthetic_corpus(seed=1)
    explainer = XgbExplainer.from_model(XgbMatchModel.fit(X, y))
    row = X.iloc[[0]]
    contributions = explainer.explain_row(row, class_index=CLASS_HOME)
    assert isinstance(contributions, list)
    assert len(contributions) == len(DEFAULT_FEATURE_COLUMNS)
    assert {c.feature for c in contributions} == set(DEFAULT_FEATURE_COLUMNS)
    assert all(isinstance(c, FeatureContribution) for c in contributions)


def test_explain_row_sorted_by_absolute_contribution() -> None:
    X, y = _synthetic_corpus(seed=2)
    explainer = XgbExplainer.from_model(XgbMatchModel.fit(X, y))
    contributions = explainer.explain_row(X.iloc[[0]], class_index=CLASS_HOME)
    abs_values = [abs(c.contribution) for c in contributions]
    assert abs_values == sorted(abs_values, reverse=True)


def test_top_features_returns_n_rows() -> None:
    X, y = _synthetic_corpus(seed=3)
    explainer = XgbExplainer.from_model(XgbMatchModel.fit(X, y))
    top = explainer.top_features(X.iloc[[0]], class_index=CLASS_HOME, n=3)
    assert len(top) == 3
    abs_values = [abs(c.contribution) for c in top]
    assert abs_values == sorted(abs_values, reverse=True)


def test_explain_row_works_with_dict_input() -> None:
    X, y = _synthetic_corpus(seed=4)
    explainer = XgbExplainer.from_model(XgbMatchModel.fit(X, y))
    row_dict = X.iloc[0].to_dict()
    contributions = explainer.explain_row(row_dict, class_index=CLASS_HOME)
    assert len(contributions) == len(DEFAULT_FEATURE_COLUMNS)


def test_explain_row_marks_missing_value_as_none() -> None:
    X, y = _synthetic_corpus(seed=5)
    explainer = XgbExplainer.from_model(XgbMatchModel.fit(X, y))
    row = X.iloc[[0]].copy()
    row["rest_days_diff"] = np.nan
    contributions = explainer.explain_row(row, class_index=CLASS_HOME)
    rest_contrib = next(c for c in contributions if c.feature == "rest_days_diff")
    assert rest_contrib.value is None


def test_explain_row_rejects_unknown_class_index() -> None:
    X, y = _synthetic_corpus()
    explainer = XgbExplainer.from_model(XgbMatchModel.fit(X, y))
    with pytest.raises(ValueError, match="class_index"):
        explainer.explain_row(X.iloc[[0]], class_index=99)


def test_explain_row_handles_missing_columns_with_nan_fill() -> None:
    X, y = _synthetic_corpus(seed=6)
    explainer = XgbExplainer.from_model(XgbMatchModel.fit(X, y))
    sparse = X.iloc[[0]].drop(columns=["xg_form_diff", "squad_age_diff"])
    contributions = explainer.explain_row(sparse, class_index=CLASS_HOME)
    # Should still cover the full feature space.
    assert {c.feature for c in contributions} == set(DEFAULT_FEATURE_COLUMNS)
    missing = next(c for c in contributions if c.feature == "xg_form_diff")
    assert missing.value is None
