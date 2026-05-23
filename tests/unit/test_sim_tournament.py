"""Tests for the top-level tournament simulator + Monte Carlo loop."""

from __future__ import annotations

import numpy as np
import pytest

from wc2026.ingest.kaggle_intl import load_scheduled
from wc2026.sim.fixtures import GROUP_LETTERS, WC2026Fixtures, parse_wc2026_fixtures
from wc2026.sim.tournament import (
    ROUND_COLUMNS,
    simulate_tournament,
    simulate_tournament_monte_carlo,
)


@pytest.fixture(scope="module")
def real_fixtures() -> WC2026Fixtures:
    return parse_wc2026_fixtures(load_scheduled())


class _DesignatedWinnerModel:
    """Mock model: each group's "designated winner" beats every other team 1-0,
    games between non-designated teams end 0-0. In knockouts of two designated
    teams, 50/50."""

    def __init__(self, designated: set[str]) -> None:
        self._designated = set(designated)

    def score_probs(self, home: str, away: str, *, neutral: bool = False) -> np.ndarray:
        _ = neutral
        p = np.zeros((11, 11))
        h_strong = home in self._designated
        a_strong = away in self._designated
        if h_strong and not a_strong:
            p[1, 0] = 1.0
        elif a_strong and not h_strong:
            p[0, 1] = 1.0
        elif h_strong and a_strong:
            p[1, 0] = 0.5
            p[0, 1] = 0.5
        else:
            # both weak → always 0-0 (forces lots / random tiebreaking)
            p[0, 0] = 1.0
        return p

    def expected_goals(self, home: str, away: str, *, neutral: bool = False) -> tuple[float, float]:
        _ = home, away, neutral
        return 1.5, 1.5


# --- single-tournament tests -----------------------------------------------


def test_single_tournament_runs_end_to_end(real_fixtures: WC2026Fixtures) -> None:
    designated = {real_fixtures.groups[letter][0] for letter in GROUP_LETTERS}
    model = _DesignatedWinnerModel(designated)
    rng = np.random.default_rng(0)
    result = simulate_tournament(real_fixtures, model, rng)  # type: ignore[arg-type]
    # Group stage: 12 results
    assert len(result.group_results) == 12
    # Third-place ranking: 12 entries
    assert len(result.third_place_ranking) == 12
    # R32: 16 matchups
    assert len(result.r32_matchups) == 16
    # Knockout results: 16 R32 + 8 R16 + 4 QF + 2 SF + 1 final = 31 matches
    assert len(result.knockout_results) == 31
    # Champion comes from the final
    assert result.champion in real_fixtures.teams


def test_designated_winners_always_win_their_group(real_fixtures: WC2026Fixtures) -> None:
    """If team X is designated as the strongest in their group (always beats their
    group rivals 1-0), they must finish 1st in every simulation."""
    designated_per_group = {letter: real_fixtures.groups[letter][0] for letter in GROUP_LETTERS}
    designated = set(designated_per_group.values())
    model = _DesignatedWinnerModel(designated)
    rng = np.random.default_rng(0)
    for _ in range(20):
        result = simulate_tournament(real_fixtures, model, rng)  # type: ignore[arg-type]
        for letter, expected_winner in designated_per_group.items():
            assert result.group_results[letter].standings[0].team == expected_winner, (
                f"in group {letter}, expected {expected_winner!r} to win"
            )


# --- monte-carlo tests -----------------------------------------------------


def test_monte_carlo_returns_probability_table_with_expected_shape(
    real_fixtures: WC2026Fixtures,
) -> None:
    designated = {real_fixtures.groups[letter][0] for letter in GROUP_LETTERS}
    model = _DesignatedWinnerModel(designated)
    summary = simulate_tournament_monte_carlo(real_fixtures, model, n_sims=10, seed=0)  # type: ignore[arg-type]
    assert summary.n_sims == 10
    assert list(summary.probabilities.columns) == list(ROUND_COLUMNS)
    # 48 teams
    assert len(summary.probabilities) == 48
    # All probabilities in [0, 1]
    assert (summary.probabilities >= 0.0).to_numpy().all()
    assert (summary.probabilities <= 1.0).to_numpy().all()


def test_monte_carlo_designated_winners_always_advance(
    real_fixtures: WC2026Fixtures,
) -> None:
    """All 12 designated winners must advance from the group stage (probability 1.0)
    and reach R32 (probability 1.0)."""
    designated_per_group = {letter: real_fixtures.groups[letter][0] for letter in GROUP_LETTERS}
    designated = set(designated_per_group.values())
    model = _DesignatedWinnerModel(designated)
    summary = simulate_tournament_monte_carlo(real_fixtures, model, n_sims=50, seed=0)  # type: ignore[arg-type]
    for letter, team in designated_per_group.items():
        assert summary.probabilities.loc[team, "group_winner"] == 1.0, (
            f"{team} (group {letter}) should win group with prob 1.0"
        )
        assert summary.probabilities.loc[team, "r32_reached"] == 1.0


def test_monte_carlo_total_advancements_sum_correctly(
    real_fixtures: WC2026Fixtures,
) -> None:
    """Across all teams, sum of probabilities for each round must equal the number
    of slots in that round."""
    designated = {real_fixtures.groups[letter][0] for letter in GROUP_LETTERS}
    model = _DesignatedWinnerModel(designated)
    summary = simulate_tournament_monte_carlo(real_fixtures, model, n_sims=30, seed=0)  # type: ignore[arg-type]
    sums = summary.probabilities.sum()
    assert sums["group_winner"] == pytest.approx(12)
    assert sums["runner_up"] == pytest.approx(12)
    assert sums["third_advance"] == pytest.approx(8)
    assert sums["r32_reached"] == pytest.approx(32)
    assert sums["r16_reached"] == pytest.approx(16)
    assert sums["qf_reached"] == pytest.approx(8)
    assert sums["sf_reached"] == pytest.approx(4)
    assert sums["final_reached"] == pytest.approx(2)
    assert sums["champion"] == pytest.approx(1)


def test_monte_carlo_rejects_non_positive_n_sims(real_fixtures: WC2026Fixtures) -> None:
    designated = {real_fixtures.groups[letter][0] for letter in GROUP_LETTERS}
    model = _DesignatedWinnerModel(designated)
    with pytest.raises(ValueError, match="n_sims must be positive"):
        simulate_tournament_monte_carlo(real_fixtures, model, n_sims=0)  # type: ignore[arg-type]
