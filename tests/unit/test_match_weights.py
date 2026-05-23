"""Unit tests for match weighting (time decay + tournament importance)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from wc2026.features.match_weights import (
    K_CONTINENTAL_FINAL,
    K_FRIENDLY,
    K_OTHER_TOURNAMENT,
    K_QUALIFIER_OR_MAJOR,
    K_WORLD_CUP_FINAL,
    combined_weight,
    match_importance_weight,
    time_decay_weight,
)

# --- importance --------------------------------------------------------------


@pytest.mark.parametrize(
    ("tournament", "expected"),
    [
        ("FIFA World Cup", K_WORLD_CUP_FINAL),
        ("Friendly", K_FRIENDLY),
        ("FIFA World Cup qualification", K_QUALIFIER_OR_MAJOR),
        ("UEFA Euro qualification", K_QUALIFIER_OR_MAJOR),
        ("African Cup of Nations qualification", K_QUALIFIER_OR_MAJOR),
        ("UEFA Nations League", K_QUALIFIER_OR_MAJOR),
        ("CONCACAF Nations League", K_QUALIFIER_OR_MAJOR),
        ("UEFA Euro", K_CONTINENTAL_FINAL),
        ("Copa América", K_CONTINENTAL_FINAL),
        ("Copa America", K_CONTINENTAL_FINAL),
        ("African Cup of Nations", K_CONTINENTAL_FINAL),
        ("AFC Asian Cup", K_CONTINENTAL_FINAL),
        ("Gold Cup", K_CONTINENTAL_FINAL),
        ("FIFA Confederations Cup", K_CONTINENTAL_FINAL),
        ("Gulf Cup", K_OTHER_TOURNAMENT),
        ("CECAFA Cup", K_OTHER_TOURNAMENT),
        ("Merdeka Tournament", K_OTHER_TOURNAMENT),
        ("", K_OTHER_TOURNAMENT),
        (None, K_OTHER_TOURNAMENT),
    ],
)
def test_match_importance_weight(tournament: str | None, expected: int) -> None:
    assert match_importance_weight(tournament) == expected


def test_qualifier_check_runs_before_continental_check() -> None:
    # "UEFA Euro qualification" starts with "UEFA Euro" — must NOT be classified as a final.
    assert match_importance_weight("UEFA Euro qualification") == K_QUALIFIER_OR_MAJOR


# --- time decay --------------------------------------------------------------


def test_time_decay_scalar_today_is_one() -> None:
    today = pd.Timestamp("2026-05-23")
    assert time_decay_weight(today, today, half_life_days=730) == 1.0


def test_time_decay_scalar_one_half_life_is_one_half() -> None:
    ref = pd.Timestamp("2026-05-23")
    past = ref - pd.Timedelta(days=730)
    assert math.isclose(time_decay_weight(past, ref, half_life_days=730), 0.5, rel_tol=1e-9)


def test_time_decay_scalar_two_half_lives_is_one_quarter() -> None:
    ref = pd.Timestamp("2026-05-23")
    past = ref - pd.Timedelta(days=1460)
    assert math.isclose(time_decay_weight(past, ref, half_life_days=730), 0.25, rel_tol=1e-9)


def test_time_decay_future_is_clamped_to_today() -> None:
    ref = pd.Timestamp("2026-05-23")
    future = ref + pd.Timedelta(days=100)
    # negative age clamps to 0 → weight = 1.0
    assert time_decay_weight(future, ref, half_life_days=730) == 1.0


def test_time_decay_series() -> None:
    ref = pd.Timestamp("2026-05-23")
    dates = pd.Series(pd.to_datetime(["2026-05-23", "2024-05-24", "2022-05-25", "2018-05-26"]))
    weights = time_decay_weight(dates, ref, half_life_days=730)
    # today → 1.0
    assert math.isclose(weights.iloc[0], 1.0, rel_tol=1e-9)
    # ~2y ago → ~0.5
    assert math.isclose(weights.iloc[1], 0.5, rel_tol=1e-3)
    # ~4y ago → ~0.25
    assert math.isclose(weights.iloc[2], 0.25, rel_tol=1e-3)
    # ~8y ago → ~0.0625
    assert math.isclose(weights.iloc[3], 0.0625, rel_tol=1e-3)
    # strictly decreasing
    assert (np.diff(weights.to_numpy()) < 0).all()


def test_time_decay_invalid_half_life() -> None:
    with pytest.raises(ValueError, match="half_life_days must be positive"):
        time_decay_weight(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-05-23"), half_life_days=0)


# --- combined ----------------------------------------------------------------


def test_combined_weight_multiplies_decay_and_importance() -> None:
    ref = pd.Timestamp("2026-05-23")
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-23", "2026-05-23"]),
            "tournament": ["FIFA World Cup", "Friendly"],
            "home_team": ["A", "C"],
            "away_team": ["B", "D"],
        }
    )
    w = combined_weight(df, ref_date=ref, half_life_days=730)
    # both have age=0 → decay 1.0; only importance differs
    assert math.isclose(w.iloc[0], float(K_WORLD_CUP_FINAL), rel_tol=1e-9)
    assert math.isclose(w.iloc[1], float(K_FRIENDLY), rel_tol=1e-9)


def test_combined_weight_old_world_cup_outweighs_recent_friendly_when_decay_short() -> None:
    """A WC match (K=60) is 3x as important as a Friendly (K=20). With a 2y half-life
    the WC match must be less than 2-half-lives old (≈4y) to still outweigh today's friendly.
    Pick exactly 2 half-lives → WC weight = 60*0.25=15, Friendly = 20 → friendly wins.
    """
    ref = pd.Timestamp("2026-05-23")
    four_years_ago = ref - pd.Timedelta(days=1460)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime([four_years_ago, ref]),
            "tournament": ["FIFA World Cup", "Friendly"],
            "home_team": ["A", "C"],
            "away_team": ["B", "D"],
        }
    )
    w = combined_weight(df, ref_date=ref, half_life_days=730)
    assert math.isclose(w.iloc[0], 60 * 0.25, rel_tol=1e-3)
    assert math.isclose(w.iloc[1], 20.0, rel_tol=1e-9)
    assert w.iloc[1] > w.iloc[0]


def test_combined_weight_requires_columns() -> None:
    df = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"])})
    with pytest.raises(ValueError, match="'date' and 'tournament'"):
        combined_weight(df, ref_date=pd.Timestamp("2026-05-23"), half_life_days=730)
