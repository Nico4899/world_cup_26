"""Group-stage simulation: draw scorelines from the model, apply FIFA tiebreakers.

Tiebreaker order
----------------
Implemented order, used here for ranking the four teams in each group:

    1. Points (3 W, 1 D, 0 L)
    2. Overall goal difference
    3. Overall goals scored
    4. Head-to-head points (among the currently tied subset only)
    5. Head-to-head goal difference
    6. Head-to-head goals scored
    7. Conduct score      — NOT IMPLEMENTED; needs yellow/red-card data
    8. FIFA ranking       — used if provided, else skipped
    9. Drawing of lots    — random, seeded by the rng argument

This matches the official Russia 2018 and Qatar 2022 procedures. The 2026
regulations document should be cross-checked before public launch — earlier
World Cup formats put H2H criteria BEFORE overall GD/GS, and FIFA has been
known to vary the order across competitions. Once verified, the
``TIEBREAKER_VERIFIED_FOR`` constant below should be bumped to 2026.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from wc2026.models.poisson_dc import PoissonDC
from wc2026.sim.fixtures import FixtureMatch

POINTS_WIN = 3
POINTS_DRAW = 1

# Bump to "2026" once the official regulations have been cross-checked.
TIEBREAKER_VERIFIED_FOR: str = "2022"


@dataclass(frozen=True)
class GroupMatchResult:
    home_team: str
    away_team: str
    home_score: int
    away_score: int

    @property
    def winner(self) -> str | None:
        if self.home_score > self.away_score:
            return self.home_team
        if self.home_score < self.away_score:
            return self.away_team
        return None  # draw


@dataclass(frozen=True)
class TeamStanding:
    team: str
    matches: int
    wins: int
    draws: int
    losses: int
    points: int
    goals_for: int
    goals_against: int

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against


@dataclass(frozen=True)
class GroupResult:
    group: str
    matches: tuple[GroupMatchResult, ...]
    standings: tuple[TeamStanding, ...]  # ranked 1st..4th


# --- scoreline sampling ----------------------------------------------------


def sample_scoreline(
    prob_matrix: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Sample (home_goals, away_goals) from a joint probability matrix.

    The matrix is shape (M, M) where rows index home goals 0..M-1 and columns
    index away goals. Uses cumulative-sum + searchsorted to be tolerant of
    sub-epsilon float-normalisation drift.
    """
    flat = prob_matrix.ravel()
    cum = np.cumsum(flat)
    # Renormalise the final entry up to 1.0 in case of tiny truncation drift.
    cum = cum / cum[-1]
    u = rng.random()
    idx = int(np.searchsorted(cum, u, side="right"))
    idx = min(idx, flat.size - 1)
    m = prob_matrix.shape[0]
    return idx // m, idx % m


def simulate_group_matches(
    fixtures: Iterable[FixtureMatch],
    model: PoissonDC,
    rng: np.random.Generator,
) -> tuple[GroupMatchResult, ...]:
    """Sample a scoreline for each fixture using model.score_probs."""
    out: list[GroupMatchResult] = []
    for m in fixtures:
        prob = model.score_probs(m.home_team, m.away_team, neutral=m.neutral)
        h_g, a_g = sample_scoreline(prob, rng)
        out.append(
            GroupMatchResult(
                home_team=m.home_team,
                away_team=m.away_team,
                home_score=h_g,
                away_score=a_g,
            )
        )
    return tuple(out)


# --- standings + ranking ---------------------------------------------------


def compute_standings(
    teams: Sequence[str],
    matches: Iterable[GroupMatchResult],
) -> dict[str, TeamStanding]:
    """Aggregate W/D/L/points/GF/GA per team. No ranking."""
    stats: dict[str, dict[str, int]] = {
        t: {"matches": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0} for t in teams
    }
    for m in matches:
        h = stats[m.home_team]
        a = stats[m.away_team]
        h["matches"] += 1
        a["matches"] += 1
        h["gf"] += m.home_score
        h["ga"] += m.away_score
        a["gf"] += m.away_score
        a["ga"] += m.home_score
        if m.home_score > m.away_score:
            h["wins"] += 1
            a["losses"] += 1
        elif m.home_score < m.away_score:
            a["wins"] += 1
            h["losses"] += 1
        else:
            h["draws"] += 1
            a["draws"] += 1
    return {
        t: TeamStanding(
            team=t,
            matches=v["matches"],
            wins=v["wins"],
            draws=v["draws"],
            losses=v["losses"],
            points=v["wins"] * POINTS_WIN + v["draws"] * POINTS_DRAW,
            goals_for=v["gf"],
            goals_against=v["ga"],
        )
        for t, v in stats.items()
    }


def _primary_key(standing: TeamStanding) -> tuple[int, int, int]:
    """Sort key for the first three FIFA tiebreakers (negate for descending)."""
    return (-standing.points, -standing.goal_difference, -standing.goals_for)


def _h2h_key(
    team: str,
    matches: Iterable[GroupMatchResult],
    among: set[str],
) -> tuple[int, int, int]:
    """Compute the head-to-head key restricted to matches between `among` teams."""
    relevant = [
        m
        for m in matches
        if m.home_team in among and m.away_team in among and (team in (m.home_team, m.away_team))
    ]
    standings = compute_standings(sorted(among), relevant)
    s = standings[team]
    return (-s.points, -s.goal_difference, -s.goals_for)


def rank_teams(
    teams: Sequence[str],
    matches: Sequence[GroupMatchResult],
    rng: np.random.Generator,
    *,
    fifa_ranking: dict[str, int] | None = None,
) -> tuple[TeamStanding, ...]:
    """Apply the tiebreaker chain and return standings in finishing order (1st first)."""
    standings = compute_standings(teams, matches)
    fifa = fifa_ranking or {}

    # First pass: sort by primary key. Then group by ties and break them.
    primary_sorted = sorted(teams, key=lambda t: _primary_key(standings[t]))

    # Group consecutive equal-primary-key teams.
    out: list[str] = []
    i = 0
    while i < len(primary_sorted):
        j = i + 1
        key_i = _primary_key(standings[primary_sorted[i]])
        while j < len(primary_sorted) and _primary_key(standings[primary_sorted[j]]) == key_i:
            j += 1
        tied = primary_sorted[i:j]
        if len(tied) == 1:
            out.append(tied[0])
        else:
            out.extend(_break_ties(tied, matches, standings, fifa, rng))
        i = j

    return tuple(standings[t] for t in out)


def _break_ties(
    tied: Sequence[str],
    all_matches: Sequence[GroupMatchResult],
    standings: dict[str, TeamStanding],
    fifa_ranking: dict[str, int],
    rng: np.random.Generator,
) -> list[str]:
    """Break ties among `tied` using H2H → FIFA ranking → drawing of lots."""
    among = set(tied)

    # H2H sub-ranking
    h2h_sorted = sorted(tied, key=lambda t: _h2h_key(t, all_matches, among))
    out: list[str] = []
    i = 0
    while i < len(h2h_sorted):
        j = i + 1
        key_i = _h2h_key(h2h_sorted[i], all_matches, among)
        while j < len(h2h_sorted) and _h2h_key(h2h_sorted[j], all_matches, among) == key_i:
            j += 1
        still_tied = h2h_sorted[i:j]
        if len(still_tied) == 1:
            out.append(still_tied[0])
        else:
            # FIFA ranking (lower = better; teams without an entry sort last)
            fifa_sorted = sorted(still_tied, key=lambda t: fifa_ranking.get(t, 10**9))
            # If FIFA also tied or unavailable, draw lots (random)
            out.extend(
                sorted(fifa_sorted, key=lambda t: (fifa_ranking.get(t, 10**9), rng.random()))
            )
            _ = standings  # silence unused-arg if FIFA path doesn't read it
        i = j
    return out


def simulate_group(
    group_letter: str,
    teams: Sequence[str],
    fixtures: Sequence[FixtureMatch],
    model: PoissonDC,
    rng: np.random.Generator,
    *,
    fifa_ranking: dict[str, int] | None = None,
) -> GroupResult:
    """One full group simulation: sample 6 scorelines, rank with tiebreakers."""
    if len(teams) != 4:
        raise ValueError(f"group must have 4 teams, got {len(teams)}")
    if len(fixtures) != 6:
        raise ValueError(f"group must have 6 fixtures, got {len(fixtures)}")
    if any(m.group != group_letter for m in fixtures):
        raise ValueError(f"fixture group mismatch for group {group_letter}")
    matches = simulate_group_matches(fixtures, model, rng)
    standings = rank_teams(teams, matches, rng, fifa_ranking=fifa_ranking)
    return GroupResult(group=group_letter, matches=matches, standings=standings)
