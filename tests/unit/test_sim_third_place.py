"""Tests for the best-third-placed cross-group ranker."""

from __future__ import annotations

import string

import numpy as np
import pytest

from wc2026.sim.groups import GroupResult, TeamStanding
from wc2026.sim.third_place import N_ADVANCING, N_THIRDS, rank_third_placed


def _stub_group(
    letter: str,
    third_team: str,
    *,
    points: int,
    gd: int,
    gf: int,
) -> GroupResult:
    """Make a stub GroupResult where the 3rd-placed team has the given stats."""
    return GroupResult(
        group=letter,
        matches=(),
        standings=(
            TeamStanding(f"{letter}_1st", 3, 3, 0, 0, 9, 5, 0),
            TeamStanding(f"{letter}_2nd", 3, 2, 0, 1, 6, 4, 2),
            TeamStanding(
                third_team,
                3,
                points // 3,  # rough W/D split (doesn't matter for the test)
                points % 3,
                3 - points // 3 - points % 3,
                points,
                gf,
                gf - gd,
            ),
            TeamStanding(f"{letter}_4th", 3, 0, 0, 3, 0, 0, 7),
        ),
    )


def _twelve_distinct_thirds(stats: list[tuple[int, int, int]]) -> dict[str, GroupResult]:
    """stats is a list of (points, gd, gf) for the third-placed team in groups A..L."""
    assert len(stats) == N_THIRDS
    letters = string.ascii_uppercase[:N_THIRDS]
    return {
        letter: _stub_group(letter, f"third_{letter}", points=p, gd=gd, gf=gf)
        for letter, (p, gd, gf) in zip(letters, stats, strict=True)
    }


def test_rank_third_placed_orders_by_points_then_gd_then_gs() -> None:
    rng = np.random.default_rng(0)
    stats = [
        (4, 0, 3),  # A
        (5, 1, 4),  # B
        (5, 2, 4),  # C — higher GD than B
        (5, 2, 5),  # D — same as C but more GF
        (3, -1, 2),  # E
        (3, -2, 2),  # F — lower GD than E
        (1, -3, 1),  # G
        (1, -4, 1),  # H
        (2, 0, 1),  # I
        (4, -1, 2),  # J
        (0, -5, 0),  # K
        (6, 3, 6),  # L — highest
    ]
    ranking = rank_third_placed(_twelve_distinct_thirds(stats), rng)
    teams = [e.team for e in ranking]
    # L (6pt, +3) should be first.
    assert teams[0] == "third_L"
    # Next: D > C > B (all 5pt, ordered by GD then GS)
    assert teams[1] == "third_D"
    assert teams[2] == "third_C"
    assert teams[3] == "third_B"
    # E > F (both 3pt, E has higher GD)
    e_idx = teams.index("third_E")
    f_idx = teams.index("third_F")
    assert e_idx < f_idx


def test_rank_third_placed_returns_twelve_entries_with_ranks_one_to_twelve() -> None:
    rng = np.random.default_rng(0)
    stats = [(4, 0, 3)] * N_THIRDS  # all identical — random ranking
    ranking = rank_third_placed(_twelve_distinct_thirds(stats), rng)
    assert len(ranking) == N_THIRDS
    assert [e.rank for e in ranking] == list(range(1, N_THIRDS + 1))


def test_top_eight_advance() -> None:
    rng = np.random.default_rng(0)
    # 8 teams clearly above 4 teams
    stats = [(6, 3, 6)] * N_ADVANCING + [(1, -3, 1)] * (N_THIRDS - N_ADVANCING)
    ranking = rank_third_placed(_twelve_distinct_thirds(stats), rng)
    advancing_teams = {e.team for e in ranking[:N_ADVANCING]}
    eliminated_teams = {e.team for e in ranking[N_ADVANCING:]}
    # the 8 high-stat teams (from groups A..H) all advance
    assert advancing_teams == {f"third_{letter}" for letter in string.ascii_uppercase[:N_ADVANCING]}
    assert eliminated_teams == {
        f"third_{letter}" for letter in string.ascii_uppercase[N_ADVANCING:N_THIRDS]
    }


def test_fifa_ranking_breaks_ties_before_random() -> None:
    rng = np.random.default_rng(0)
    stats = [(4, 0, 3)] * N_THIRDS  # everyone tied on primary
    # Build FIFA ranking where third_A is best (rank 1), third_L is worst (rank 12).
    fifa = {f"third_{letter}": i for i, letter in enumerate(string.ascii_uppercase[:N_THIRDS], 1)}
    ranking = rank_third_placed(_twelve_distinct_thirds(stats), rng, fifa_ranking=fifa)
    teams = [e.team for e in ranking]
    # FIFA ranking should fully determine the order since primary keys are all equal
    expected = [f"third_{letter}" for letter in string.ascii_uppercase[:N_THIRDS]]
    assert teams == expected


def test_random_lots_when_fully_tied_over_many_seeds() -> None:
    """If FIFA ranking isn't provided AND primary is tied, drawing of lots decides;
    each team should land in 1st place roughly 1/N of the time across many seeds."""
    stats = [(4, 0, 3)] * N_THIRDS
    first_counts: dict[str, int] = {
        f"third_{letter}": 0 for letter in string.ascii_uppercase[:N_THIRDS]
    }
    n_trials = 1200
    for seed in range(n_trials):
        ranking = rank_third_placed(_twelve_distinct_thirds(stats), np.random.default_rng(seed))
        first_counts[ranking[0].team] += 1
    # expect ~100 each +/- ~30 (std for binomial(1200, 1/12) ~= sqrt(91.7) ~= 9.6, 3*std ~= 29)
    expected = n_trials / N_THIRDS
    for team, count in first_counts.items():
        assert abs(count - expected) < 40, f"{team} got 1st {count} times, expected ~{expected:.0f}"


def test_rank_rejects_wrong_group_count() -> None:
    rng = np.random.default_rng(0)
    short = _twelve_distinct_thirds([(4, 0, 3)] * N_THIRDS)
    short.pop("A")
    with pytest.raises(ValueError, match="expected 12 group results"):
        rank_third_placed(short, rng)


def test_rank_rejects_group_with_wrong_standings_length() -> None:
    rng = np.random.default_rng(0)
    full = _twelve_distinct_thirds([(4, 0, 3)] * N_THIRDS)
    full["A"] = GroupResult(group="A", matches=(), standings=full["A"].standings[:3])  # only 3
    with pytest.raises(ValueError, match=r"group A has 3 standings"):
        rank_third_placed(full, rng)
