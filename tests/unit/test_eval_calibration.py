"""Tests for calibration metrics."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from wc2026.eval.calibration import (
    aggregate,
    base_rates,
    baseline_log_loss,
    match_brier,
    match_log_loss,
    match_rps,
    observed_outcome,
    reliability_diagram,
)


def test_observed_outcome() -> None:
    assert observed_outcome(2, 1) == "H"
    assert observed_outcome(0, 1) == "A"
    assert observed_outcome(1, 1) == "D"
    assert observed_outcome(0, 0) == "D"


def test_match_log_loss_perfect_forecast_is_zero() -> None:
    assert match_log_loss("H", p_home=1.0, p_draw=0.0, p_away=0.0) == pytest.approx(0.0)


def test_match_log_loss_uniform_forecast_is_log3() -> None:
    ll = match_log_loss("D", p_home=1 / 3, p_draw=1 / 3, p_away=1 / 3)
    assert ll == pytest.approx(math.log(3))


def test_match_log_loss_misses_lower_bound() -> None:
    # observed had prob 0 → forced to EPS, ll is large but finite
    ll = match_log_loss("H", p_home=0.0, p_draw=0.5, p_away=0.5)
    assert ll > 30  # -log(1e-15) ≈ 34.5
    assert math.isfinite(ll)


def test_match_brier_perfect_is_zero() -> None:
    assert match_brier("H", p_home=1.0, p_draw=0.0, p_away=0.0) == pytest.approx(0.0)


def test_match_brier_uniform() -> None:
    b = match_brier("H", p_home=1 / 3, p_draw=1 / 3, p_away=1 / 3)
    # (1/3 - 1)^2 + (1/3)^2 + (1/3)^2 = 4/9 + 1/9 + 1/9 = 6/9 = 2/3
    assert b == pytest.approx(2 / 3)


def test_match_rps_perfect_is_zero() -> None:
    assert match_rps("H", p_home=1.0, p_draw=0.0, p_away=0.0) == pytest.approx(0.0)
    assert match_rps("D", p_home=0.0, p_draw=1.0, p_away=0.0) == pytest.approx(0.0)
    assert match_rps("A", p_home=0.0, p_draw=0.0, p_away=1.0) == pytest.approx(0.0)


def test_match_rps_penalises_distance_to_observed() -> None:
    # Observed = H. Putting all mass on Away is worse than putting all on Draw.
    rps_draw = match_rps("H", p_home=0.0, p_draw=1.0, p_away=0.0)
    rps_away = match_rps("H", p_home=0.0, p_draw=0.0, p_away=1.0)
    assert rps_away > rps_draw


def test_aggregate_basic() -> None:
    df = pd.DataFrame(
        [
            {"observed": "H", "p_home": 1.0, "p_draw": 0.0, "p_away": 0.0},  # perfect
            {"observed": "D", "p_home": 0.5, "p_draw": 0.3, "p_away": 0.2},  # log(0.3) ~ 1.20
        ]
    )
    m = aggregate(df)
    assert m.n == 2
    # log_loss mean = (0 + -log(0.3)) / 2
    assert m.log_loss == pytest.approx((0 + (-math.log(0.3))) / 2)


def test_aggregate_empty() -> None:
    m = aggregate(pd.DataFrame(columns=["observed", "p_home", "p_draw", "p_away"]))
    assert m.n == 0
    assert math.isnan(m.log_loss)


def test_aggregate_missing_columns_raises() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        aggregate(pd.DataFrame({"observed": ["H"]}))


def test_reliability_diagram_bin_counts_and_means() -> None:
    df = pd.DataFrame(
        [
            {"observed": "H", "p_home": 0.05, "p_draw": 0.5, "p_away": 0.45},
            {
                "observed": "H",
                "p_home": 0.95,
                "p_draw": 0.03,
                "p_away": 0.02,
            },  # confident H, correct
            {"observed": "A", "p_home": 0.92, "p_draw": 0.05, "p_away": 0.03},  # confident H, wrong
            {"observed": "D", "p_home": 0.35, "p_draw": 0.40, "p_away": 0.25},
        ]
    )
    bins = reliability_diagram(df, n_bins=10)
    # H outcome, top bin (0.9, 1.0): 2 obs (the two confident H predictions), realized freq = 1/2
    top_bin_h = next(b for b in bins if b.outcome == "H" and b.bin_low >= 0.9)
    assert top_bin_h.n == 2
    assert top_bin_h.realized_frequency == pytest.approx(0.5)
    # Total count per outcome equals n
    total_h = sum(b.n for b in bins if b.outcome == "H")
    assert total_h == 4


def test_baseline_log_loss_equals_entropy_of_base_rates() -> None:
    # All H → entropy 0
    assert baseline_log_loss(["H"] * 10) == pytest.approx(0.0)
    # Uniform → log(3)
    assert baseline_log_loss(["H", "D", "A"]) == pytest.approx(math.log(3))


def test_base_rates() -> None:
    obs = ["H", "H", "H", "D", "A"]
    rates = base_rates(obs)
    assert rates == {"H": 0.6, "D": 0.2, "A": 0.2}
