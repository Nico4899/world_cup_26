"""Unit tests for the logistic-regression xG shot model."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wc2026.models.xg_shot_model import (
    DEFAULT_BODY_PART_LEVELS,
    DEFAULT_PATTERN_LEVELS,
    XgShotModel,
    attach_predicted_xg,
)


def _synthetic_corpus(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Synthetic shot corpus where p(goal) decreases sharply with distance.

    Close shots (≤ 8m) are gold; long shots (≥ 20m) are mostly misses; mid-
    range follows. Just enough signal for logistic regression to learn the
    coefficients we expect.
    """
    rng = np.random.default_rng(seed)
    dist = rng.uniform(2.0, 36.0, size=n)
    angle = rng.uniform(0.1, 1.6, size=n)
    body_part = rng.choice(DEFAULT_BODY_PART_LEVELS, size=n, p=[0.55, 0.30, 0.10, 0.05])
    # Force some penalties + free kicks so the model can learn those bumps.
    pattern = rng.choice(
        DEFAULT_PATTERN_LEVELS,
        size=n,
        p=[0.80, 0.07, 0.04, 0.08, 0.01],
    )
    # Probability decreases with distance, increases with angle, +0.6 for penalties.
    logits = (
        4.0
        - 0.30 * dist
        + 1.2 * angle
        + np.where(pattern == "Penalty", 2.5, 0.0)
        - 0.3 * (body_part == "Head").astype(float)
    )
    p_goal = 1.0 / (1.0 + np.exp(-logits))
    is_goal = rng.random(size=n) < p_goal
    return pd.DataFrame(
        {
            "distance_to_goal": dist,
            "angle_to_goal": angle,
            "body_part": body_part,
            "pattern_of_play": pattern,
            "is_goal": is_goal,
        }
    )


def test_fit_returns_model_with_expected_feature_names() -> None:
    model = XgShotModel.fit(_synthetic_corpus())
    # distance + angle + inv_distance + (4 body_part levels - 1 ref) + (5 patterns - 1 ref) = 10
    expected_count = 3 + (len(DEFAULT_BODY_PART_LEVELS) - 1) + (len(DEFAULT_PATTERN_LEVELS) - 1)
    assert len(model.feature_names) == expected_count
    assert "distance_to_goal" in model.feature_names
    assert "angle_to_goal" in model.feature_names
    assert "inv_distance" in model.feature_names
    assert "pat_penalty" in model.feature_names


def test_distance_coefficient_is_negative() -> None:
    """Closer shots → higher p(goal): the distance coefficient must be negative."""
    model = XgShotModel.fit(_synthetic_corpus(n=2000))
    coef = dict(zip(model.feature_names, model.coefficients, strict=True))
    assert coef["distance_to_goal"] < 0


def test_penalty_coefficient_is_positive() -> None:
    """Penalties have a large positive bump on goal probability."""
    model = XgShotModel.fit(_synthetic_corpus(n=2000))
    coef = dict(zip(model.feature_names, model.coefficients, strict=True))
    assert coef["pat_penalty"] > 0


def test_predict_proba_returns_probabilities_in_unit_interval() -> None:
    model = XgShotModel.fit(_synthetic_corpus())
    df = _synthetic_corpus(n=64, seed=42)
    p = model.predict_proba(df)
    assert p.shape == (64,)
    assert np.all((p >= 0.0) & (p <= 1.0))


def test_predict_proba_close_central_shot_higher_than_far_wide() -> None:
    model = XgShotModel.fit(_synthetic_corpus(n=2000, seed=1))
    df = pd.DataFrame(
        {
            "distance_to_goal": [3.0, 28.0],
            "angle_to_goal": [1.4, 0.15],
            "body_part": ["Right Foot", "Right Foot"],
            "pattern_of_play": ["Open Play", "Open Play"],
        }
    )
    p = model.predict_proba(df)
    assert p[0] > p[1]


def test_predict_proba_penalty_close_to_published_rate() -> None:
    """Empirical penalty conversion is ~0.76. A model trained on data with a
    +2.5 logit bump for penalties should predict at least 0.55."""
    model = XgShotModel.fit(_synthetic_corpus(n=2000, seed=2))
    df = pd.DataFrame(
        {
            "distance_to_goal": [11.0],
            "angle_to_goal": [0.5],
            "body_part": ["Right Foot"],
            "pattern_of_play": ["Penalty"],
        }
    )
    p = model.predict_proba(df)[0]
    assert p > 0.55


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    model = XgShotModel.fit(_synthetic_corpus(seed=3))
    out = model.save(tmp_path / "model.json")
    loaded = XgShotModel.load(out)
    assert loaded.intercept == model.intercept
    assert loaded.feature_names == model.feature_names
    assert loaded.coefficients == model.coefficients


def test_predict_proba_after_load_matches_pre_save(tmp_path: Path) -> None:
    """Save/load must be bit-perfect for probabilities at typical inputs."""
    model = XgShotModel.fit(_synthetic_corpus(seed=4))
    out = model.save(tmp_path / "model.json")
    loaded = XgShotModel.load(out)
    df = _synthetic_corpus(n=10, seed=99)
    p_before = model.predict_proba(df)
    p_after = loaded.predict_proba(df)
    assert np.allclose(p_before, p_after)


def test_fit_raises_when_required_column_missing() -> None:
    bad = pd.DataFrame(
        {
            "distance_to_goal": [10.0],
            "angle_to_goal": [0.5],
            "body_part": ["Right Foot"],
            # pattern_of_play missing
            "is_goal": [True],
        }
    )
    with pytest.raises(ValueError, match="missing"):
        XgShotModel.fit(bad)


def test_predict_handles_unknown_category_via_reference_fallback() -> None:
    model = XgShotModel.fit(_synthetic_corpus())
    df = pd.DataFrame(
        {
            "distance_to_goal": [10.0],
            "angle_to_goal": [0.5],
            "body_part": ["No Such Body Part"],
            "pattern_of_play": ["No Such Pattern"],
        }
    )
    # Should not crash; unknown categories collapse to the reference level.
    p = model.predict_proba(df)
    assert 0.0 < p[0] < 1.0
    assert not math.isnan(p[0])


def test_attach_predicted_xg_adds_column() -> None:
    model = XgShotModel.fit(_synthetic_corpus(seed=5))
    df = _synthetic_corpus(n=20, seed=6)
    out = attach_predicted_xg(df, model)
    assert "our_xg" in out.columns
    assert out["our_xg"].between(0, 1).all()
