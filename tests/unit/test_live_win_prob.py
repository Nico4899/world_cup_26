"""Unit tests for the live in-match win-probability model."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wc2026.models.live_win_prob import (
    DEFAULT_FEATURE_COLUMNS,
    LiveWinProbModel,
    minutes_remaining_from_minute,
)
from wc2026.models.xgb_classifier import CLASS_AWAY, CLASS_DRAW, CLASS_HOME


def _synthetic_state_corpus(n: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate state snapshots whose label depends mostly on (goal_diff, minutes_remaining)."""
    rng = np.random.default_rng(seed)
    elo_diff = rng.normal(0.0, 100.0, size=n)
    goal_diff = rng.integers(-3, 4, size=n)
    minutes_remaining = rng.integers(0, 90, size=n)
    red_diff = rng.choice([-1, 0, 0, 0, 0, 1], size=n)
    # Logit: late-game lead is decisive; early on, Elo matters more.
    time_lock = np.where(minutes_remaining < 20, 3.5, 1.0)
    logits_h = 0.012 * elo_diff + time_lock * goal_diff - 0.6 * red_diff
    logits_a = -0.012 * elo_diff - time_lock * goal_diff + 0.6 * red_diff
    noise = rng.normal(0, 1.0, size=n)
    y = np.where(
        logits_h + noise > 1.0,
        CLASS_HOME,
        np.where(logits_a + noise > 1.0, CLASS_AWAY, CLASS_DRAW),
    ).astype(int)
    X = pd.DataFrame(
        {
            "elo_diff": elo_diff,
            "goal_diff": goal_diff,
            "minutes_remaining": minutes_remaining,
            "red_diff": red_diff,
        }
    )
    return X, y


def test_minutes_remaining_regulation() -> None:
    assert minutes_remaining_from_minute(0) == 90
    assert minutes_remaining_from_minute(45, period=1) == 45
    assert minutes_remaining_from_minute(90, period=2) == 0


def test_minutes_remaining_extra_time() -> None:
    assert minutes_remaining_from_minute(95, period=3) == 25
    assert minutes_remaining_from_minute(120, period=4) == 0


def test_minutes_remaining_penalties_returns_zero() -> None:
    assert minutes_remaining_from_minute(120, period=5) == 0


def test_fit_emits_three_class_coefficients() -> None:
    X, y = _synthetic_state_corpus()
    model = LiveWinProbModel.fit(X, y)
    assert len(model.intercepts) == 3
    assert len(model.coefficients) == 3
    for row in model.coefficients:
        assert len(row) == len(DEFAULT_FEATURE_COLUMNS)


def test_predict_proba_sums_to_one_per_row() -> None:
    X, y = _synthetic_state_corpus()
    model = LiveWinProbModel.fit(X, y)
    probs = model.predict_proba(X.head(20))
    assert probs.shape == (20, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-9)


def test_predict_proba_lead_dominates_late() -> None:
    """A 2-goal lead with 5 minutes left → home_win > 0.85."""
    X, y = _synthetic_state_corpus(n=4000, seed=1)
    model = LiveWinProbModel.fit(X, y)
    probs = model.predict_one(elo_diff=0, goal_diff=2, minutes_remaining=5, red_diff=0)
    assert probs["home_win"] > 0.85


def test_predict_proba_responds_to_red_card() -> None:
    """An equal-Elo, scoreless mid-match state shifts toward the side with the man advantage."""
    X, y = _synthetic_state_corpus(n=4000, seed=2)
    model = LiveWinProbModel.fit(X, y)
    no_red = model.predict_one(elo_diff=0, goal_diff=0, minutes_remaining=45, red_diff=0)
    home_advantage = model.predict_one(elo_diff=0, goal_diff=0, minutes_remaining=45, red_diff=-1)
    # red_diff=-1 means the AWAY side has a man down → home_win should rise.
    assert home_advantage["home_win"] > no_red["home_win"]


def test_predict_proba_responds_to_elo_diff_early() -> None:
    X, y = _synthetic_state_corpus(n=4000, seed=3)
    model = LiveWinProbModel.fit(X, y)
    weak = model.predict_one(elo_diff=-200, goal_diff=0, minutes_remaining=85, red_diff=0)
    strong = model.predict_one(elo_diff=200, goal_diff=0, minutes_remaining=85, red_diff=0)
    assert strong["home_win"] > weak["home_win"] + 0.1


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    X, y = _synthetic_state_corpus(seed=4)
    model = LiveWinProbModel.fit(X, y)
    out = model.save(tmp_path / "model.json")
    loaded = LiveWinProbModel.load(out)
    assert loaded.intercepts == model.intercepts
    assert loaded.coefficients == model.coefficients
    p1 = model.predict_proba(X.head(10))
    p2 = loaded.predict_proba(X.head(10))
    assert np.allclose(p1, p2)


def test_fit_rejects_missing_column() -> None:
    X, y = _synthetic_state_corpus()
    bad = X.drop(columns=["red_diff"])
    with pytest.raises(ValueError, match="missing required"):
        LiveWinProbModel.fit(bad, y)


def test_fit_rejects_invalid_label() -> None:
    X, _ = _synthetic_state_corpus()
    y_bad = np.full(len(X), 99, dtype=int)
    with pytest.raises(ValueError, match="labels must be a subset"):
        LiveWinProbModel.fit(X, y_bad)


def test_predict_one_outputs_dict_summing_to_one() -> None:
    X, y = _synthetic_state_corpus(seed=5)
    model = LiveWinProbModel.fit(X, y)
    probs = model.predict_one(elo_diff=50, goal_diff=1, minutes_remaining=30, red_diff=0)
    assert set(probs.keys()) == {"home_win", "draw", "away_win"}
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-9)
