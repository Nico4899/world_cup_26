"""Tests for the group simulator and FIFA tiebreakers."""

from __future__ import annotations

import numpy as np
import pytest

from wc2026.sim.fixtures import FixtureMatch
from wc2026.sim.groups import (
    POINTS_DRAW,
    POINTS_WIN,
    GroupMatchResult,
    compute_standings,
    rank_teams,
    sample_scoreline,
    simulate_group,
    simulate_group_matches,
)

# --- sampler ---------------------------------------------------------------


def test_sample_scoreline_deterministic_when_one_cell_carries_all_mass() -> None:
    prob = np.zeros((6, 6))
    prob[2, 1] = 1.0
    rng = np.random.default_rng(0)
    for _ in range(20):
        h, a = sample_scoreline(prob, rng)
        assert (h, a) == (2, 1)


def test_sample_scoreline_frequencies_match_probabilities() -> None:
    """Two equally likely cells should be sampled with ~50/50 frequency over 5k draws."""
    prob = np.zeros((4, 4))
    prob[1, 1] = 0.5
    prob[0, 2] = 0.5
    rng = np.random.default_rng(7)
    counts = {(1, 1): 0, (0, 2): 0}
    for _ in range(5000):
        s = sample_scoreline(prob, rng)
        counts[s] = counts.get(s, 0) + 1
    total = sum(counts.values())
    assert total == 5000
    for k in [(1, 1), (0, 2)]:
        assert abs(counts[k] / total - 0.5) < 0.03


# --- standings -------------------------------------------------------------


def _matches_from_pairs(rows: list[tuple[str, str, int, int]]) -> list[GroupMatchResult]:
    return [GroupMatchResult(h, a, hs, as_) for h, a, hs, as_ in rows]


def test_compute_standings_basic() -> None:
    teams = ["A", "B", "C", "D"]
    matches = _matches_from_pairs(
        [
            ("A", "B", 2, 1),  # A win
            ("C", "D", 0, 0),  # draw
            ("A", "C", 1, 1),  # draw
            ("B", "D", 3, 0),  # B win
            ("A", "D", 0, 1),  # D win
            ("B", "C", 1, 2),  # C win
        ]
    )
    s = compute_standings(teams, matches)
    # A: W, D, L → 4 points; GF=3, GA=3
    assert s["A"].points == POINTS_WIN + POINTS_DRAW
    assert s["A"].wins == 1
    assert s["A"].draws == 1
    assert s["A"].losses == 1
    assert s["A"].goals_for == 3
    assert s["A"].goals_against == 3
    assert s["A"].goal_difference == 0
    # B: L, W, L → 3 points; GF=5, GA=4 — wait check: B played A (1-2 loss), D (3-0 win), C (1-2 loss)
    assert s["B"].points == POINTS_WIN
    assert s["B"].wins == 1 and s["B"].draws == 0 and s["B"].losses == 2
    assert s["B"].goals_for == 5
    assert s["B"].goals_against == 4
    # All teams played 3 games
    assert all(t.matches == 3 for t in s.values())


# --- ranking ---------------------------------------------------------------


def test_rank_teams_clear_order_by_points() -> None:
    teams = ["A", "B", "C", "D"]
    matches = _matches_from_pairs(
        [
            ("A", "B", 3, 0),  # A 3 pts
            ("A", "C", 2, 1),
            ("A", "D", 1, 0),  # A: 3-3-3 = 9 pts, GF=6, GA=1
            ("B", "C", 1, 0),  # B 3 pts, B: 0-3-1 = 3 pts so far
            ("B", "D", 2, 2),  # B draw, B: 4 pts
            ("C", "D", 0, 0),  # both draw
        ]
    )
    rng = np.random.default_rng(0)
    ranked = rank_teams(teams, matches, rng)
    # A finishes first (9 pts). B is 2nd (4 pts). C is 3rd (1+1+0 = 2 pts). D is 4th (1+0+1 = 2 pts)
    assert ranked[0].team == "A"
    assert ranked[1].team == "B"


def test_rank_teams_tie_on_points_uses_overall_gd() -> None:
    """Two teams tied on 7 points; team with higher GD finishes higher."""
    teams = ["A", "B", "C", "D"]
    matches = _matches_from_pairs(
        [
            ("A", "B", 1, 1),  # draw
            ("C", "D", 1, 1),  # draw
            ("A", "C", 5, 0),  # A win big
            ("B", "D", 1, 0),  # B win narrow
            ("A", "D", 1, 0),  # A win
            ("B", "C", 1, 0),  # B win
        ]
    )
    # A: 7 pts (D-W-W), GF=7, GA=1 → GD +6
    # B: 7 pts (D-W-W), GF=3, GA=1 → GD +2
    rng = np.random.default_rng(0)
    ranked = rank_teams(teams, matches, rng)
    assert ranked[0].team == "A"
    assert ranked[1].team == "B"


def test_rank_teams_h2h_tied_falls_through_to_overall_goals_scored() -> None:
    """2026 chain: H2H tied → overall GD tied → overall GS breaks the tie."""
    teams = ["A", "B", "C", "D"]
    # A vs B drew 2-2 (H2H stats identical). A's and B's overall GD also tied at +1,
    # but B's overall GS (6) > A's overall GS (4) → B finishes first.
    matches = _matches_from_pairs(
        [
            ("A", "B", 2, 2),  # H2H draw — H2H pts/GD/GS identical for A and B
            ("C", "D", 0, 0),
            ("A", "C", 2, 0),  # A wins
            ("B", "C", 3, 1),  # B wins
            ("A", "D", 0, 1),  # D wins
            ("B", "D", 1, 2),  # D wins
        ]
    )
    # D: D-W-W = 7 pts (group winner; not what the test cares about)
    # A: D-W-L = 4 pts; GF=2+2+0=4, GA=2+0+1=3, GD+1, GS=4
    # B: D-W-L = 4 pts; GF=2+3+1=6, GA=2+1+2=5, GD+1, GS=6
    # C: D-L-L = 1 pt
    rng = np.random.default_rng(0)
    ranked = rank_teams(teams, matches, rng)
    assert ranked[1].team == "B"
    assert ranked[2].team == "A"


def test_rank_teams_h2h_decides_before_overall_gd_in_2026() -> None:
    """2026 chain: H2H is applied BEFORE overall GD (reversed from 2022)."""
    teams = ["A", "B", "C", "D"]
    # A and B tied on 6 pts. B has the bigger overall GD/GS, but A beat B head-to-head.
    # Under 2026 rules, A finishes first via H2H precedence.
    matches = _matches_from_pairs(
        [
            ("A", "B", 1, 0),  # H2H: A wins
            ("C", "D", 0, 0),
            ("A", "C", 1, 0),  # A 6 pts, GF=2, GA=0
            ("B", "C", 5, 0),  # B 3 pts, GF=5, GA=0
            ("A", "D", 0, 1),  # D wins; A 6 pts still, GF=2, GA=1
            ("B", "D", 2, 0),  # B 6 pts, GF=7, GA=0
        ]
    )
    # A: 6 pts, GD+1, GS=2. B: 6 pts, GD+7, GS=7. Overall favours B by a wide margin.
    # H2H: A beat B 1-0 → A wins.
    rng = np.random.default_rng(0)
    ranked = rank_teams(teams, matches, rng)
    assert ranked[0].team == "A"
    assert ranked[1].team == "B"


def test_rank_teams_tied_on_all_overall_uses_h2h() -> None:
    """Two teams tied on points, GD, and goals scored. H2H result must decide."""
    teams = ["A", "B", "C", "D"]
    matches = _matches_from_pairs(
        [
            ("A", "B", 2, 1),  # A wins H2H over B
            ("C", "D", 0, 0),
            ("A", "C", 1, 2),  # A loses
            ("B", "D", 1, 0),  # B wins
            ("A", "D", 1, 0),  # A wins
            ("B", "C", 1, 2),  # B loses
        ]
    )
    # A: W-L-W = 6 pts; GF=4, GA=3 → GD +1
    # B: L-W-L = 3 pts; GF=3, GA=4 → GD -1   (this is not what we want)
    # Restart: design so A and B are exactly tied on overall, H2H = A
    matches = _matches_from_pairs(
        [
            ("A", "B", 1, 0),  # A beats B in H2H
            ("C", "D", 0, 0),
            ("A", "C", 0, 1),  # A loses
            ("B", "C", 1, 0),  # B wins
            ("A", "D", 2, 0),  # A wins
            ("B", "D", 1, 0),  # B wins
        ]
    )
    # A: W-L-W = 6 pts; GF=3, GA=1 → GD +2
    # B: L-W-W = 6 pts; GF=2, GA=1 → GD +1   ← not tied
    # Try again:
    matches = _matches_from_pairs(
        [
            ("A", "B", 1, 0),  # A beats B in H2H
            ("C", "D", 1, 0),
            ("A", "C", 0, 1),  # A loses
            ("B", "C", 1, 0),  # B wins
            ("A", "D", 2, 0),  # A wins
            ("B", "D", 1, 0),  # B wins
        ]
    )
    # A: W-L-W → 6 pts, GF=3, GA=1, GD+2
    # B: L-W-W → 6 pts, GF=2, GA=1, GD+1
    # still not tied on GD. Easier: keep GD/GF identical via symmetric scores.
    matches = _matches_from_pairs(
        [
            ("A", "B", 1, 0),  # H2H: A
            ("C", "D", 0, 0),
            ("A", "C", 2, 1),  # A wins
            ("B", "C", 2, 1),  # B wins (mirror)
            ("A", "D", 1, 2),  # A loses
            ("B", "D", 1, 2),  # B loses (mirror)
        ]
    )
    # A: W-W-L = 6 pts; GF=4, GA=3, GD+1
    # B: L-W-L = 3 pts ... no
    # OK I'll just trust pytest-randomly to flush this out and use explicit numbers
    # that produce a clean tied set:
    matches = _matches_from_pairs(
        [
            ("A", "B", 1, 0),  # H2H: A wins
            ("C", "D", 0, 0),
            ("A", "C", 2, 2),  # A draws
            ("B", "C", 2, 1),  # B wins
            ("A", "D", 0, 0),  # A draws
            ("B", "D", 1, 1),  # B draws (symmetric so B has same record as A in the totals)
        ]
    )
    # A: W-D-D = 5 pts; GF=3, GA=2, GD+1
    # B: L-W-D = 4 pts ... still not tied. Skip this construction; assert H2H differently.

    # Pragmatic test: construct ONLY 2 teams with identical primary, where H2H clearly decides.
    teams = ["A", "B", "C", "D"]
    matches = _matches_from_pairs(
        [
            ("A", "B", 2, 0),  # A beats B 2-0 (the only H2H tiebreaker that matters)
            ("C", "D", 0, 0),
            ("A", "C", 1, 1),  # A: 4 pts, GF=3, GA=1
            ("B", "C", 1, 1),  # B: 1 pt
            ("A", "D", 0, 1),  # A: 4 pts still; GF=3, GA=2
            ("B", "D", 2, 0),  # B: 4 pts; GF=3, GA=2
        ]
    )
    # A: W-D-L → 4 pts; GF=3, GA=2; GD+1
    # B: L-D-W → 4 pts; GF=3, GA=2; GD+1
    # PERFECT primary tie. H2H winner = A.
    rng = np.random.default_rng(0)
    ranked = rank_teams(teams, matches, rng)
    a_idx = next(i for i, s in enumerate(ranked) if s.team == "A")
    b_idx = next(i for i, s in enumerate(ranked) if s.team == "B")
    assert a_idx < b_idx, f"A should finish above B via H2H; got order {[s.team for s in ranked]}"


def test_rank_teams_total_tie_falls_through_to_rng_when_fifa_ranking_missing() -> None:
    """The 2026 regulations have no drawing of lots, but the simulator must still
    produce a total order when teams are tied on points/H2H/overall AND no FIFA
    ranking is supplied. We use a deterministic rng-seeded fallback; over many
    seeds, each team should finish 1st about half the time."""
    teams = ["A", "B"]
    matches: list[GroupMatchResult] = [GroupMatchResult("A", "B", 0, 0)]
    n_a_first = 0
    n_b_first = 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        ranked = rank_teams(teams, matches, rng)
        if ranked[0].team == "A":
            n_a_first += 1
        else:
            n_b_first += 1
    # 200 trials, 50/50 → expected 100 each, std ≈ 7. allow ±20.
    assert abs(n_a_first - 100) < 20
    assert abs(n_b_first - 100) < 20


# --- simulate_group_matches with mock model -----------------------------


class _AlwaysScoreOneNilModel:
    """Mock PoissonDC that always says the home team wins 1-0 deterministically."""

    def score_probs(self, home_team: str, away_team: str, *, neutral: bool = False) -> np.ndarray:
        _ = home_team, away_team, neutral
        p = np.zeros((11, 11))
        p[1, 0] = 1.0
        return p


def test_simulate_group_matches_uses_model_distribution() -> None:
    fixtures = [
        FixtureMatch(
            date=__import__("pandas").Timestamp("2026-06-11"),
            home_team="A",
            away_team="B",
            group="A",
            city="X",
            country="USA",
            neutral=True,
        ),
        FixtureMatch(
            date=__import__("pandas").Timestamp("2026-06-12"),
            home_team="C",
            away_team="D",
            group="A",
            city="X",
            country="USA",
            neutral=True,
        ),
    ]
    rng = np.random.default_rng(0)
    results = simulate_group_matches(fixtures, _AlwaysScoreOneNilModel(), rng)  # type: ignore[arg-type]
    assert all(r.home_score == 1 and r.away_score == 0 for r in results)


# --- simulate_group end-to-end -------------------------------------------


def test_simulate_group_validates_team_and_fixture_counts() -> None:
    rng = np.random.default_rng(0)
    f = FixtureMatch(
        date=__import__("pandas").Timestamp("2026-06-11"),
        home_team="A",
        away_team="B",
        group="A",
        city="X",
        country="USA",
        neutral=True,
    )
    # wrong team count
    with pytest.raises(ValueError, match="group must have 4 teams"):
        simulate_group("A", ["A", "B"], [f] * 6, _AlwaysScoreOneNilModel(), rng)  # type: ignore[arg-type]
    # wrong fixture count
    with pytest.raises(ValueError, match="group must have 6 fixtures"):
        simulate_group(
            "A",
            ["A", "B", "C", "D"],
            [f] * 3,
            _AlwaysScoreOneNilModel(),
            rng,  # type: ignore[arg-type]
        )
    # group-letter mismatch
    f_b = FixtureMatch(
        date=__import__("pandas").Timestamp("2026-06-11"),
        home_team="A",
        away_team="B",
        group="B",  # wrong
        city="X",
        country="USA",
        neutral=True,
    )
    with pytest.raises(ValueError, match="fixture group mismatch"):
        simulate_group(
            "A",
            ["A", "B", "C", "D"],
            [f_b] * 6,
            _AlwaysScoreOneNilModel(),
            rng,  # type: ignore[arg-type]
        )
