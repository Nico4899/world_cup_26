"""Top-level Monte Carlo simulator for the 2026 World Cup.

Runs N full tournaments end-to-end (group stage → third-place ranking →
R32 → R16 → QF → SF → final), and aggregates per-team probabilities of
reaching each round.

For the bracket structure see ``bracket.py``; for group tiebreakers see
``groups.py``; for the third-place ranker see ``third_place.py``; for the
knockout single-match simulator see ``knockout.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wc2026.models.poisson_dc import PoissonDC
from wc2026.sim.bracket import (
    FINAL_PAIR,
    QF_PAIRS,
    R16_PAIRS,
    SF_PAIRS,
    resolve_r32_matchups,
)
from wc2026.sim.fixtures import GROUP_LETTERS, WC2026Fixtures
from wc2026.sim.groups import GroupResult, simulate_group
from wc2026.sim.knockout import KnockoutOutcome, ShootoutStrategy, simulate_knockout_match
from wc2026.sim.third_place import ThirdPlacedEntry, rank_third_placed

# Columns of the aggregated probability table.
#
# ``third_out`` and ``fourth`` were added to support the 5-segment group-stage
# bars on the dashboard (1st / 2nd / 3rd→R32 / 3rd-out / 4th). Older persisted
# runs that don't carry these columns degrade to a single "eliminated" bucket
# in the loader.
ROUND_COLUMNS: tuple[str, ...] = (
    "group_winner",
    "runner_up",
    "third_advance",
    "third_out",
    "fourth",
    "r32_reached",
    "r16_reached",
    "qf_reached",
    "sf_reached",
    "final_reached",
    "champion",
)


@dataclass(frozen=True)
class TournamentResult:
    """One full tournament simulation."""

    group_results: dict[str, GroupResult]
    third_place_ranking: tuple[ThirdPlacedEntry, ...]
    r32_matchups: dict[int, tuple[str, str]]
    knockout_results: dict[int, KnockoutOutcome]
    champion: str


def simulate_tournament(
    fixtures: WC2026Fixtures,
    model: PoissonDC,
    rng: np.random.Generator,
    *,
    fifa_ranking: dict[str, int] | None = None,
    shootout_strategy: ShootoutStrategy | None = None,
    known_group_results: dict[tuple[str, str], tuple[int, int]] | None = None,
    known_knockout_winners: dict[int, str] | None = None,
) -> TournamentResult:
    """Run a single tournament simulation end-to-end.

    ``shootout_strategy``, when provided, replaces the 50/50 coin-flip on every
    shootout (R32 → final). Pass an Elo-based logistic from
    :mod:`wc2026.models.shootout` for slightly better than chance shootout calls.

    ``known_group_results`` (Phase 8) locks in scorelines for group-stage
    fixtures that have already happened — keyed by ``(home_team, away_team)``.

    ``known_knockout_winners`` (Tier 3 "lock a match" feature) locks the
    *winner* of specific knockout match ids. When the locked team is one of
    the two sides that actually reached the match in this simulation, the
    lock fires (a deterministic 1-0 regulation result is recorded so the
    bracket structure is well-defined). If the locked team isn't a
    contestant, the lock is silently skipped for that sim — by design, so
    users can lock optimistic scenarios without rejecting the entire MC pass.
    """
    # --- group stage ---
    group_results: dict[str, GroupResult] = {}
    for letter in GROUP_LETTERS:
        group_results[letter] = simulate_group(
            letter,
            fixtures.groups[letter],
            fixtures.matches_in_group(letter),
            model,
            rng,
            fifa_ranking=fifa_ranking,
            known_results=known_group_results,
        )

    winners = {letter: gr.standings[0].team for letter, gr in group_results.items()}
    runners = {letter: gr.standings[1].team for letter, gr in group_results.items()}

    # --- third-place ranking + select top 8 ---
    third_ranking = rank_third_placed(group_results, rng, fifa_ranking=fifa_ranking)
    advancing_thirds = {entry.group: entry.team for entry in third_ranking[:8]}

    # --- resolve R32 matchups ---
    r32_matchups = resolve_r32_matchups(winners, runners, advancing_thirds)

    # --- knockout rounds ---
    knockout_results: dict[int, KnockoutOutcome] = {}
    winners_by_match: dict[int, str] = {}
    locked = known_knockout_winners or {}

    def _ko(home: str, away: str, *, match_id: int) -> KnockoutOutcome:
        # Apply per-match-id lock when the chosen winner is actually one of
        # the two sides; otherwise fall through to the sampler so the bracket
        # stays consistent on sims where the lock isn't reachable.
        forced = locked.get(match_id)
        if forced in (home, away):
            return KnockoutOutcome(
                home_team=home,
                away_team=away,
                winner=forced,
                regulation_score=(1, 0) if forced == home else (0, 1),
                extra_time_score=None,
                shootout_winner=None,
                decided_in="regulation",
            )
        return simulate_knockout_match(
            home, away, model, rng, neutral=True, shootout_strategy=shootout_strategy
        )

    # R32
    for mid, (a, b) in r32_matchups.items():
        out = _ko(a, b, match_id=mid)
        knockout_results[mid] = out
        winners_by_match[mid] = out.winner

    # R16, QF, SF
    for mid, ma, mb in R16_PAIRS:
        out = _ko(winners_by_match[ma], winners_by_match[mb], match_id=mid)
        knockout_results[mid] = out
        winners_by_match[mid] = out.winner
    for mid, ma, mb in QF_PAIRS:
        out = _ko(winners_by_match[ma], winners_by_match[mb], match_id=mid)
        knockout_results[mid] = out
        winners_by_match[mid] = out.winner
    for mid, ma, mb in SF_PAIRS:
        out = _ko(winners_by_match[ma], winners_by_match[mb], match_id=mid)
        knockout_results[mid] = out
        winners_by_match[mid] = out.winner

    # Final
    final_id, sf1, sf2 = FINAL_PAIR
    out = _ko(winners_by_match[sf1], winners_by_match[sf2], match_id=final_id)
    knockout_results[final_id] = out
    champion = out.winner

    return TournamentResult(
        group_results=group_results,
        third_place_ranking=third_ranking,
        r32_matchups=r32_matchups,
        knockout_results=knockout_results,
        champion=champion,
    )


@dataclass(frozen=True)
class TournamentSummary:
    """Aggregated per-team advancement probabilities across N tournament sims."""

    n_sims: int
    probabilities: pd.DataFrame  # index=team, columns=ROUND_COLUMNS

    def top(self, column: str, n: int = 10) -> pd.DataFrame:
        return self.probabilities.sort_values(column, ascending=False).head(n)


def simulate_tournament_monte_carlo(
    fixtures: WC2026Fixtures,
    model: PoissonDC,
    *,
    n_sims: int = 10_000,
    seed: int = 42,
    fifa_ranking: dict[str, int] | None = None,
    shootout_strategy: ShootoutStrategy | None = None,
    known_group_results: dict[tuple[str, str], tuple[int, int]] | None = None,
    known_knockout_winners: dict[int, str] | None = None,
) -> TournamentSummary:
    """Run ``n_sims`` simulations; return per-team advancement probabilities.

    ``known_group_results`` (Phase 8) is forwarded to every individual
    simulation, so the same locked-in scorelines apply to all ``n_sims``
    iterations. ``known_knockout_winners`` (Tier 3) does the same for
    knockout match ids — useful for "what if Team X wins the SF?" scenarios.
    """
    if n_sims <= 0:
        raise ValueError(f"n_sims must be positive, got {n_sims}")
    teams = list(fixtures.teams)
    counters: dict[str, dict[str, int]] = {t: {col: 0 for col in ROUND_COLUMNS} for t in teams}

    rng = np.random.default_rng(seed)
    for _ in range(n_sims):
        result = simulate_tournament(
            fixtures,
            model,
            rng,
            fifa_ranking=fifa_ranking,
            shootout_strategy=shootout_strategy,
            known_group_results=known_group_results,
            known_knockout_winners=known_knockout_winners,
        )
        _update_counters_from_result(counters, result)

    df = pd.DataFrame.from_dict(counters, orient="index", columns=list(ROUND_COLUMNS))
    df = df / n_sims
    df.index.name = "team"
    return TournamentSummary(n_sims=n_sims, probabilities=df)


# Bracket-progression mapping: each round records which match_ids are "reached"
# by which team. A team is in R16 iff they won their R32 match; QF iff they won
# their R16; etc.
_R32_MATCH_IDS = tuple(range(73, 89))
_R16_MATCH_IDS = tuple(mid for mid, _, _ in R16_PAIRS)
_QF_MATCH_IDS = tuple(mid for mid, _, _ in QF_PAIRS)
_SF_MATCH_IDS = tuple(mid for mid, _, _ in SF_PAIRS)

# Round-by-round match-id buckets keyed by the same labels the dashboard uses.
PATH_ROUND_MATCH_IDS: dict[str, tuple[int, ...]] = {
    "r32": _R32_MATCH_IDS,
    "r16": _R16_MATCH_IDS,
    "qf": _QF_MATCH_IDS,
    "sf": _SF_MATCH_IDS,
    "final": (FINAL_PAIR[0],),
}


def compute_path_to_final(
    fixtures: WC2026Fixtures,
    model: PoissonDC,
    *,
    n_sims: int = 2000,
    seed: int = 42,
    fifa_ranking: dict[str, int] | None = None,
    shootout_strategy: ShootoutStrategy | None = None,
    known_group_results: dict[tuple[str, str], tuple[int, int]] | None = None,
) -> dict[str, dict[str, dict[str, int]]]:
    """Histogram of per-team opponents at each knockout round.

    Returns ``{team: {round_label: {opponent: count}}}``. Each ``count`` is
    the number of simulated tournaments in which ``team`` faced ``opponent``
    in ``round_label``. Sums to ``count(team reached round)`` per (team, round).

    Computed in a single Monte Carlo pass so the Team Profile route can serve
    "most likely opponent at each stage" for any team without re-simulating.
    """
    if n_sims <= 0:
        raise ValueError(f"n_sims must be positive, got {n_sims}")
    teams = list(fixtures.teams)
    histograms: dict[str, dict[str, dict[str, int]]] = {
        t: {label: {} for label in PATH_ROUND_MATCH_IDS} for t in teams
    }

    rng = np.random.default_rng(seed)
    for _ in range(n_sims):
        result = simulate_tournament(
            fixtures,
            model,
            rng,
            fifa_ranking=fifa_ranking,
            shootout_strategy=shootout_strategy,
            known_group_results=known_group_results,
        )
        for label, mids in PATH_ROUND_MATCH_IDS.items():
            for mid in mids:
                if mid not in result.knockout_results and label != "r32":
                    continue
                if label == "r32":
                    pair = result.r32_matchups.get(mid)
                    if pair is None:
                        continue
                    a, b = pair
                    histograms[a][label][b] = histograms[a][label].get(b, 0) + 1
                    histograms[b][label][a] = histograms[b][label].get(a, 0) + 1
                else:
                    outcome = result.knockout_results[mid]
                    a, b = outcome.home_team, outcome.away_team
                    histograms[a][label][b] = histograms[a][label].get(b, 0) + 1
                    histograms[b][label][a] = histograms[b][label].get(a, 0) + 1
    return histograms


def _update_counters_from_result(
    counters: dict[str, dict[str, int]],
    result: TournamentResult,
) -> None:
    advancing_thirds_set = {e.team for e in result.third_place_ranking[:8]}

    # Group-stage finishing positions
    for gr in result.group_results.values():
        counters[gr.standings[0].team]["group_winner"] += 1
        counters[gr.standings[1].team]["runner_up"] += 1
        third_team = gr.standings[2].team
        if third_team in advancing_thirds_set:
            counters[third_team]["third_advance"] += 1
        else:
            counters[third_team]["third_out"] += 1
        counters[gr.standings[3].team]["fourth"] += 1

    # R32 reached: every team in an R32 matchup
    for a, b in result.r32_matchups.values():
        counters[a]["r32_reached"] += 1
        counters[b]["r32_reached"] += 1

    # R16/QF/SF/Final reached: the winners of the previous round
    for mid in _R32_MATCH_IDS:
        counters[result.knockout_results[mid].winner]["r16_reached"] += 1
    for mid in _R16_MATCH_IDS:
        counters[result.knockout_results[mid].winner]["qf_reached"] += 1
    for mid in _QF_MATCH_IDS:
        counters[result.knockout_results[mid].winner]["sf_reached"] += 1
    for mid in _SF_MATCH_IDS:
        counters[result.knockout_results[mid].winner]["final_reached"] += 1

    counters[result.champion]["champion"] += 1
