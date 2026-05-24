"""Tournament-level routes: aggregated Monte Carlo standings + sample bracket."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from wc2026.api.dependencies import get_fixtures, get_model
from wc2026.db.models import RawLiveEvent, TournamentSimRun, TournamentSimTeamOutcome
from wc2026.db.session import get_engine
from wc2026.ingest.football_data_org import load_wc_match_id_map
from wc2026.ingest.live_events import EVENT_FT_WHISTLE
from wc2026.models.poisson_dc import PoissonDC
from wc2026.sim.fixtures import WC2026Fixtures
from wc2026.sim.groups import POINTS_DRAW, POINTS_WIN
from wc2026.sim.knockout import ShootoutStrategy
from wc2026.sim.tournament import (
    ROUND_COLUMNS,
    TournamentResult,
    TournamentSummary,
    simulate_tournament,
    simulate_tournament_monte_carlo,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tournament")

DEFAULT_N_SIMS = 2000
DEFAULT_SEED = 42
CACHE_TTL_SECONDS = 3600  # 1 hour

# Module-level caches keyed by request params. Thread-safe via the lock; reads + writes
# are atomic enough under the GIL that the simple (timestamp, value) tuple suffices.
_STANDINGS_CACHE: dict[tuple[int, int], tuple[float, TournamentSummary]] = {}
_BRACKET_CACHE: dict[int, tuple[float, TournamentResult]] = {}
_CACHE_LOCK = threading.Lock()


def _load_persisted_summary() -> tuple[TournamentSummary, int, str] | None:
    """Return the most-recent persisted MC run as a TournamentSummary.

    The Phase 8 ``rerun_monte_carlo`` script writes per-team probabilities
    into ``tournament_sim_runs`` + ``tournament_sim_team_outcomes``. We
    rebuild ``TournamentSummary.probabilities`` from those rows; the
    ``third_advance`` column the in-process simulator emits isn't stored
    explicitly, but it's algebraically recoverable as
    ``advance_r32_p - group_winner_p - group_runner_up_p``.

    Returns ``(summary, run_id, model_version)`` or ``None`` when the table
    is empty or the DB is unreachable.
    """
    try:
        eng = get_engine()
        with Session(eng, future=True) as session:
            run = session.scalars(
                select(TournamentSimRun).order_by(desc(TournamentSimRun.created_at)).limit(1)
            ).first()
            if run is None:
                return None
            outcomes = list(
                session.scalars(
                    select(TournamentSimTeamOutcome).where(
                        TournamentSimTeamOutcome.run_id == run.run_id
                    )
                )
            )
            if not outcomes:
                return None
            # third_advance = advance_r32_p - group_winner_p - group_runner_up_p
            # is recoverable from existing columns. third_out_p + fourth_p are
            # nullable (Phase 12 added them); when null, fall back to splitting
            # the residual 50/50 so the dashboard still renders five segments
            # without misleading absolutes — old runs are clearly marked.
            def _split_eliminated(o: TournamentSimTeamOutcome) -> tuple[float, float]:
                if o.third_out_p is not None and o.fourth_p is not None:
                    return float(o.third_out_p), float(o.fourth_p)
                eliminated = max(1.0 - float(o.advance_r32_p), 0.0)
                return eliminated / 2, eliminated / 2

            split = [_split_eliminated(o) for o in outcomes]
            data = {
                "group_winner": [o.group_winner_p for o in outcomes],
                "runner_up": [o.group_runner_up_p for o in outcomes],
                "third_advance": [
                    max(o.advance_r32_p - o.group_winner_p - o.group_runner_up_p, 0.0)
                    for o in outcomes
                ],
                "third_out": [s[0] for s in split],
                "fourth": [s[1] for s in split],
                "r32_reached": [o.advance_r32_p for o in outcomes],
                "r16_reached": [o.advance_r16_p for o in outcomes],
                "qf_reached": [o.quarterfinal_p for o in outcomes],
                "sf_reached": [o.semifinal_p for o in outcomes],
                "final_reached": [o.final_p for o in outcomes],
                "champion": [o.champion_p for o in outcomes],
            }
            df = pd.DataFrame(data, index=[o.team for o in outcomes], columns=list(ROUND_COLUMNS))
            df.index.name = "team"
            return (
                TournamentSummary(n_sims=int(run.n_sims), probabilities=df),
                int(run.run_id),
                str(run.model_version),
            )
    except Exception:
        logger.debug(
            "standings: persisted-run lookup failed; falling back to in-process MC", exc_info=True
        )
        return None


def _cached_summary(
    model: PoissonDC,
    fixtures: WC2026Fixtures,
    n_sims: int,
    seed: int,
    shootout_strategy: ShootoutStrategy | None,
) -> TournamentSummary:
    # NB: cache key intentionally excludes shootout_strategy because the strategy
    # is fixed per-process (built once in the lifespan) and would not be hashable.
    key = (n_sims, seed)
    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _STANDINGS_CACHE.get(key)
        if cached is not None and now - cached[0] < CACHE_TTL_SECONDS:
            return cached[1]
    summary = simulate_tournament_monte_carlo(
        fixtures, model, n_sims=n_sims, seed=seed, shootout_strategy=shootout_strategy
    )
    with _CACHE_LOCK:
        _STANDINGS_CACHE[key] = (now, summary)
    return summary


def _cached_bracket(
    model: PoissonDC,
    fixtures: WC2026Fixtures,
    seed: int,
    shootout_strategy: ShootoutStrategy | None,
) -> TournamentResult:
    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _BRACKET_CACHE.get(seed)
        if cached is not None and now - cached[0] < CACHE_TTL_SECONDS:
            return cached[1]
    rng = np.random.default_rng(seed)
    result = simulate_tournament(fixtures, model, rng, shootout_strategy=shootout_strategy)
    with _CACHE_LOCK:
        _BRACKET_CACHE[seed] = (now, result)
    return result


def _build_group_block(letter: str, summary, fixtures: WC2026Fixtures) -> dict[str, Any]:
    teams = list(fixtures.groups[letter])
    rows = []
    for team in teams:
        row = summary.probabilities.loc[team]
        third_out = float(row.get("third_out", 0.0)) if "third_out" in row.index else 0.0
        fourth = float(row.get("fourth", 0.0)) if "fourth" in row.index else 0.0
        eliminated_total = float(
            max(0.0, 1.0 - row["group_winner"] - row["runner_up"] - row["third_advance"])
        )
        # P(eliminated) stays for back-compat callers; the new 5-segment fields
        # let the dashboard render the spec'd bars.
        rows.append(
            {
                "team": team,
                "p_first": float(row["group_winner"]),
                "p_second": float(row["runner_up"]),
                "p_third_advance": float(row["third_advance"]),
                "p_third_out": third_out,
                "p_fourth": fourth,
                "p_eliminated": eliminated_total,
            }
        )
    rows.sort(key=lambda r: r["p_first"] + r["p_second"] + r["p_third_advance"], reverse=True)
    return {"group": letter, "teams": rows}


@router.get("/standings")
def standings(
    request: Request,
    n_sims: int = Query(default=DEFAULT_N_SIMS, ge=100, le=20_000),
    seed: int = Query(default=DEFAULT_SEED),
    use_persisted: bool = Query(
        default=True,
        description=(
            "Phase 8: when True (the default) the route serves the most-recent "
            "persisted Monte Carlo run from ``tournament_sim_runs``, which the "
            "rerun_monte_carlo script writes after each completed WC 2026 match. "
            "Set False to force an in-process simulation (legacy behaviour)."
        ),
    ),
    model: PoissonDC = Depends(get_model),
    fixtures: WC2026Fixtures = Depends(get_fixtures),
) -> dict[str, Any]:
    shootout_strategy = getattr(request.app.state, "shootout_strategy", None)
    loaded = _load_persisted_summary() if use_persisted else None
    if loaded is not None:
        summary, run_id, persisted_model_version = loaded
        source = "persisted"
    else:
        summary = _cached_summary(model, fixtures, n_sims, seed, shootout_strategy)
        source = "in_process"
        run_id = None
        persisted_model_version = None
    groups = [_build_group_block(letter, summary, fixtures) for letter in fixtures.groups]
    # Top-10 championship probabilities for the headline
    top = (
        summary.probabilities.sort_values("champion", ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={"index": "team"})
    )
    headline = [
        {
            "team": r["team"],
            "p_champion": float(r["champion"]),
            "p_final": float(r["final_reached"]),
            "p_sf": float(r["sf_reached"]),
            "p_qf": float(r["qf_reached"]),
        }
        for _, r in top.iterrows()
    ]
    return {
        "n_sims": int(summary.n_sims),
        "round_columns": list(ROUND_COLUMNS),
        "groups": groups,
        "headline": headline,
        "source": source,
        "run_id": run_id,
        "model_version": persisted_model_version,
    }


def _empty_team_row(team: str) -> dict[str, Any]:
    return {
        "team": team,
        "played": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "points": 0,
        "goals_for": 0,
        "goals_against": 0,
        "goal_difference": 0,
    }


def _live_group_tally(fixtures: WC2026Fixtures) -> dict[str, list[dict[str, Any]]]:
    """Build the per-group "current table" from completed live events.

    Cross-references FDO ``match_id`` → ``(date, home, away)`` via the cached
    fixtures, picks the FT_WHISTLE row for each completed match (the poller
    persists one per finished fixture), and tallies points / GF / GA per team
    inside the team's group. Teams with no completed matches show zeros so
    the dashboard always sees a complete 4-row block per group.
    """
    blocks: dict[str, dict[str, dict[str, Any]]] = {}
    for letter, teams in fixtures.groups.items():
        blocks[letter] = {team: _empty_team_row(team) for team in teams}

    id_map = load_wc_match_id_map()
    if not id_map:
        return {letter: list(rows.values()) for letter, rows in blocks.items()}

    # Group lookup is by team; build once.
    team_to_group: dict[str, str] = {
        team: letter for letter, members in fixtures.groups.items() for team in members
    }

    try:
        eng = get_engine()
        with Session(eng, future=True) as session:
            ft_rows = list(
                session.scalars(
                    select(RawLiveEvent).where(RawLiveEvent.event_type == EVENT_FT_WHISTLE)
                )
            )
    except Exception:
        logger.debug("groups-live: DB unreachable, returning zero-tally blocks", exc_info=True)
        return {letter: list(rows.values()) for letter, rows in blocks.items()}

    for r in ft_rows:
        fixture = id_map.get(int(r.match_id))
        if fixture is None:
            continue
        _, home, away = fixture
        letter = team_to_group.get(home) or team_to_group.get(away)
        if letter is None:
            continue
        home_row = blocks[letter].get(home)
        away_row = blocks[letter].get(away)
        if home_row is None or away_row is None:
            continue
        h, a = int(r.home_score_after), int(r.away_score_after)
        for row, gf, ga in ((home_row, h, a), (away_row, a, h)):
            row["played"] += 1
            row["goals_for"] += gf
            row["goals_against"] += ga
        if h > a:
            home_row["wins"] += 1
            home_row["points"] += POINTS_WIN
            away_row["losses"] += 1
        elif h < a:
            away_row["wins"] += 1
            away_row["points"] += POINTS_WIN
            home_row["losses"] += 1
        else:
            home_row["draws"] += 1
            away_row["draws"] += 1
            home_row["points"] += POINTS_DRAW
            away_row["points"] += POINTS_DRAW

    for letter_rows in blocks.values():
        for row in letter_rows.values():
            row["goal_difference"] = row["goals_for"] - row["goals_against"]

    return {letter: list(rows.values()) for letter, rows in blocks.items()}


@router.get("/groups-live")
def groups_live(fixtures: WC2026Fixtures = Depends(get_fixtures)) -> dict[str, Any]:
    """Per-group current points + goal difference from completed live matches.

    Each block has every team in the group; ``played=0`` means the side has no
    FT_WHISTLE row yet (typical before the tournament opens). The Groups page
    renders this above the Monte Carlo bars so the spec's "current points,
    current GD" requirement is visible at a glance.
    """
    blocks = _live_group_tally(fixtures)
    out_groups = []
    for letter in fixtures.groups:
        rows = sorted(
            blocks.get(letter, []),
            key=lambda r: (r["points"], r["goal_difference"], r["goals_for"]),
            reverse=True,
        )
        out_groups.append({"group": letter, "teams": rows})
    return {"groups": out_groups}


@router.get("/bracket")
def bracket(
    request: Request,
    seed: int = Query(default=DEFAULT_SEED, description="Sample one bracket realisation."),
    model: PoissonDC = Depends(get_model),
    fixtures: WC2026Fixtures = Depends(get_fixtures),
) -> dict[str, Any]:
    shootout_strategy = getattr(request.app.state, "shootout_strategy", None)
    result = _cached_bracket(model, fixtures, seed, shootout_strategy)
    matches = []
    for mid, outcome in sorted(result.knockout_results.items()):
        # round_label by match id range per FIFA numbering
        if 73 <= mid <= 88:
            round_label = "R32"
        elif 89 <= mid <= 96:
            round_label = "R16"
        elif 97 <= mid <= 100:
            round_label = "QF"
        elif 101 <= mid <= 102:
            round_label = "SF"
        else:
            round_label = "Final"
        matches.append(
            {
                "match_id": mid,
                "round": round_label,
                "home_team": outcome.home_team,
                "away_team": outcome.away_team,
                "winner": outcome.winner,
                "decided_in": outcome.decided_in,
                "regulation_score": list(outcome.regulation_score),
            }
        )
    return {
        "seed": seed,
        "champion": result.champion,
        "matches": matches,
    }
