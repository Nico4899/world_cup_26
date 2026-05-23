"""Tests for the knockout single-match simulator (regulation → ET → shootout)."""

from __future__ import annotations

import numpy as np
import pytest

from wc2026.sim.knockout import simulate_knockout_match


class _AlwaysScoreModel:
    """Mock PoissonDC that always returns a fixed scoreline."""

    def __init__(self, home: int, away: int, lh: float = 1.5, la: float = 1.5) -> None:
        self._home = home
        self._away = away
        self._lh = lh
        self._la = la

    def score_probs(self, home_team: str, away_team: str, *, neutral: bool = False) -> np.ndarray:
        _ = home_team, away_team, neutral
        p = np.zeros((11, 11))
        p[self._home, self._away] = 1.0
        return p

    def expected_goals(
        self, home_team: str, away_team: str, *, neutral: bool = False
    ) -> tuple[float, float]:
        _ = home_team, away_team, neutral
        return self._lh, self._la


def test_regulation_winner_when_home_scores_more() -> None:
    rng = np.random.default_rng(0)
    out = simulate_knockout_match("A", "B", _AlwaysScoreModel(2, 0), rng)  # type: ignore[arg-type]
    assert out.regulation_score == (2, 0)
    assert out.extra_time_score is None
    assert out.shootout_winner is None
    assert out.winner == "A"
    assert out.decided_in == "regulation"
    assert out.total_score == (2, 0)


def test_regulation_winner_when_away_scores_more() -> None:
    rng = np.random.default_rng(0)
    out = simulate_knockout_match("A", "B", _AlwaysScoreModel(0, 3), rng)  # type: ignore[arg-type]
    assert out.winner == "B"
    assert out.decided_in == "regulation"


def test_extra_time_decides_when_regulation_tied_and_et_breaks_tie() -> None:
    """With a Tied-1-1 regulation model and very high ET goal rates, ET should
    almost always produce a winner (not a shootout). Run 200 trials; expect a
    very high fraction decided in extra_time."""
    n_et = 0
    n_shootout = 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        # ET rate is lh/3 = 5.0/3 ≈ 1.67 expected ET goals each side;
        # P(both sides have equal ET count) is small (~20%).
        out = simulate_knockout_match("A", "B", _AlwaysScoreModel(1, 1, lh=5.0, la=5.0), rng)  # type: ignore[arg-type]
        assert out.regulation_score == (1, 1)
        if out.decided_in == "extra_time":
            n_et += 1
            assert out.extra_time_score is not None
            assert out.extra_time_score[0] != out.extra_time_score[1]
            assert out.shootout_winner is None
        elif out.decided_in == "shootout":
            n_shootout += 1
            assert out.shootout_winner is not None
        else:
            raise AssertionError(f"unexpected decided_in={out.decided_in}")
    # ET should dominate
    assert n_et > n_shootout
    assert n_et + n_shootout == 200


def test_injected_shootout_strategy_overrides_coin_flip() -> None:
    """When shootout_strategy is provided, it replaces the 50/50 fallback. Verify the
    strategy is actually called (and its return value used as the winner)."""
    calls: list[tuple[str, str]] = []

    def always_home(home: str, away: str, rng: np.random.Generator) -> str:
        _ = rng
        calls.append((home, away))
        return home

    rng = np.random.default_rng(0)
    out = simulate_knockout_match(
        "A",
        "B",
        _AlwaysScoreModel(0, 0, lh=0.0, la=0.0),  # type: ignore[arg-type]
        rng,
        shootout_strategy=always_home,
    )
    assert out.decided_in == "shootout"
    assert out.winner == "A"
    assert out.shootout_winner == "A"
    assert calls == [("A", "B")]


def test_injected_shootout_strategy_must_return_a_participating_team() -> None:
    def bogus(home: str, away: str, rng: np.random.Generator) -> str:
        _ = home, away, rng
        return "Atlantis"

    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="must be either"):
        simulate_knockout_match(
            "A",
            "B",
            _AlwaysScoreModel(0, 0, lh=0.0, la=0.0),  # type: ignore[arg-type]
            rng,
            shootout_strategy=bogus,
        )


def test_shootout_50_50_when_et_can_not_break_tie() -> None:
    """If model returns 0-0 with both ET lambdas = 0, every match goes to shootout
    with 50/50 winner. Over many seeds the home/away split should be ~50/50."""
    home_wins = 0
    n_trials = 1500
    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        out = simulate_knockout_match("A", "B", _AlwaysScoreModel(0, 0, lh=0.0, la=0.0), rng)  # type: ignore[arg-type]
        assert out.decided_in == "shootout"
        assert out.regulation_score == (0, 0)
        assert out.extra_time_score == (0, 0)
        if out.winner == "A":
            home_wins += 1
    expected = n_trials / 2
    # std for binomial(1500, 0.5) = sqrt(375) = 19; 3-sigma = 57.
    assert abs(home_wins - expected) < 75


def test_total_score_aggregates_regulation_plus_et() -> None:
    """When the match goes to ET, total_score equals regulation + ET (not shootout)."""
    rng = np.random.default_rng(0)
    out = simulate_knockout_match("A", "B", _AlwaysScoreModel(1, 1, lh=5.0, la=0.1), rng)  # type: ignore[arg-type]
    assert out.regulation_score == (1, 1)
    if out.extra_time_score is not None:
        eh, ea = out.extra_time_score
        assert out.total_score == (1 + eh, 1 + ea)
