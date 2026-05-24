"""Per-team queries — recent form from the team's perspective."""

from __future__ import annotations

import logging
import threading
import time

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from wc2026.api.dependencies import get_fixtures, get_model, get_played_df
from wc2026.api.schemas import (
    EloHistoryPoint,
    FifaRankingPoint,
    PathOpponent,
    PathRound,
    SquadMember,
    TeamAssetsResponse,
    TeamEloHistory,
    TeamFifaRankingHistory,
    TeamPathToFinal,
    TeamRecentMatch,
    TeamSquadResponse,
    TeamTournamentProbabilities,
    TeamXgFormResponse,
    XgFormSplit,
)
from wc2026.db.models import (
    RawEloSnapshot,
    RawFifaRanking,
    RawMatchXg,
    RawSquad,
    RawTeamAsset,
    TournamentSimRun,
    TournamentSimTeamOutcome,
)
from wc2026.db.session import get_engine
from wc2026.models.poisson_dc import PoissonDC
from wc2026.sim.fixtures import WC2026Fixtures
from wc2026.sim.tournament import PATH_ROUND_MATCH_IDS, compute_path_to_final

router = APIRouter(prefix="/api/v1/teams")

DEFAULT_RECENT_N = 5
MAX_RECENT_N = 30

logger = logging.getLogger(__name__)


@router.get("/{team}/recent", response_model=list[TeamRecentMatch])
def recent_form(
    team: str,
    n: int = Query(default=DEFAULT_RECENT_N, ge=1, le=MAX_RECENT_N),
    played: pd.DataFrame = Depends(get_played_df),
) -> list[TeamRecentMatch]:
    """Return the most recent ``n`` matches for ``team``, framed from its perspective.

    422 if the team has never played any international (per the Jürisoo dataset).
    """
    mask = (played["home_team"] == team) | (played["away_team"] == team)
    sub = played[mask].sort_values("date", ascending=False).head(n)
    if sub.empty:
        raise HTTPException(status_code=422, detail=f"unknown team: {team!r}")

    out: list[TeamRecentMatch] = []
    for _, row in sub.iterrows():
        is_home = row["home_team"] == team
        opponent = row["away_team"] if is_home else row["home_team"]
        goals_for = int(row["home_score"] if is_home else row["away_score"])
        goals_against = int(row["away_score"] if is_home else row["home_score"])
        if goals_for > goals_against:
            result = "W"
        elif goals_for < goals_against:
            result = "L"
        else:
            result = "D"
        venue = "neutral" if bool(row["neutral"]) else ("home" if is_home else "away")
        out.append(
            TeamRecentMatch(
                date=row["date"].date(),
                opponent=opponent,
                venue=venue,
                goals_for=goals_for,
                goals_against=goals_against,
                result=result,
                tournament=str(row["tournament"]),
            )
        )
    return out


@router.get("/{team}/elo-history", response_model=TeamEloHistory)
def team_elo_history(team: str) -> TeamEloHistory:
    """Daily Elo snapshots for ``team``, oldest → newest.

    Empty ``history`` when there's no row for that team name in
    ``raw_elo_snapshots`` (the schema's team key is the eloratings.net team
    code; we join by ``team_name``). 503 only when the DB itself is unreachable.
    """
    try:
        eng = get_engine()
        with Session(eng, future=True) as session:
            rows = list(
                session.scalars(
                    select(RawEloSnapshot)
                    .where(RawEloSnapshot.team_name == team)
                    .order_by(RawEloSnapshot.snapshot_date)
                )
            )
    except Exception as exc:
        logger.debug("team_elo_history: DB error for %s", team, exc_info=True)
        raise HTTPException(
            status_code=503, detail=f"team_elo_history DB query failed: {exc.__class__.__name__}"
        ) from exc
    return TeamEloHistory(
        team=team,
        history=[
            EloHistoryPoint(
                snapshot_date=r.snapshot_date,
                rating=float(r.rating),
                global_rank=r.global_rank,
            )
            for r in rows
        ],
    )


@router.get("/{team}/tournament-probs", response_model=TeamTournamentProbabilities)
def team_tournament_probs(team: str) -> TeamTournamentProbabilities:
    """Per-team advancement probs from the most-recent persisted MC run.

    Returns an empty payload (with ``run_id=None``) when no run exists yet —
    Phase 8's ``rerun_monte_carlo`` script populates this table.
    """
    try:
        eng = get_engine()
        with Session(eng, future=True) as session:
            run = session.scalars(
                select(TournamentSimRun).order_by(TournamentSimRun.created_at.desc()).limit(1)
            ).first()
            if run is None:
                return TeamTournamentProbabilities(team=team)
            outcome = session.scalars(
                select(TournamentSimTeamOutcome).where(
                    TournamentSimTeamOutcome.run_id == run.run_id,
                    TournamentSimTeamOutcome.team == team,
                )
            ).first()
            if outcome is None:
                return TeamTournamentProbabilities(
                    team=team,
                    run_id=run.run_id,
                    n_sims=run.n_sims,
                    model_version=run.model_version,
                )
            return TeamTournamentProbabilities(
                team=team,
                run_id=run.run_id,
                n_sims=run.n_sims,
                model_version=run.model_version,
                group_winner_p=outcome.group_winner_p,
                group_runner_up_p=outcome.group_runner_up_p,
                advance_r32_p=outcome.advance_r32_p,
                advance_r16_p=outcome.advance_r16_p,
                quarterfinal_p=outcome.quarterfinal_p,
                semifinal_p=outcome.semifinal_p,
                final_p=outcome.final_p,
                champion_p=outcome.champion_p,
            )
    except Exception as exc:
        logger.debug("team_tournament_probs: DB error for %s", team, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"team_tournament_probs DB query failed: {exc.__class__.__name__}",
        ) from exc


@router.get("/{team}/fifa-rankings", response_model=TeamFifaRankingHistory)
def team_fifa_rankings(team: str) -> TeamFifaRankingHistory:
    """Monthly FIFA Men's Ranking history for ``team`` (oldest → newest).

    Empty when no rows exist; the dashboard then explains the
    ``fifa_ranking_refresh`` scheduler job hasn't populated them yet.
    """
    try:
        eng = get_engine()
        with Session(eng, future=True) as session:
            rows = list(
                session.scalars(
                    select(RawFifaRanking)
                    .where(RawFifaRanking.team == team)
                    .order_by(RawFifaRanking.ranking_date)
                )
            )
    except Exception as exc:
        logger.debug("team_fifa_rankings: DB error for %s", team, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"team_fifa_rankings DB query failed: {exc.__class__.__name__}",
        ) from exc
    return TeamFifaRankingHistory(
        team=team,
        history=[
            FifaRankingPoint(
                ranking_date=r.ranking_date,
                rank=int(r.rank),
                points=None if r.points is None else float(r.points),
                previous_rank=r.previous_rank,
            )
            for r in rows
        ],
    )


@router.get("/{team}/squad", response_model=TeamSquadResponse)
def team_squad(team: str) -> TeamSquadResponse:
    """Latest tournament-squad snapshot for ``team`` from ``raw_squads``.

    Picks the freshest ``snapshot_date`` and returns every player on it,
    sorted by shirt number when present. Empty payload when no row exists.
    """
    try:
        eng = get_engine()
        with Session(eng, future=True) as session:
            latest_row = session.execute(
                select(RawSquad.tournament, RawSquad.snapshot_date)
                .where(RawSquad.team == team)
                .order_by(RawSquad.snapshot_date.desc())
                .limit(1)
            ).first()
            if latest_row is None:
                return TeamSquadResponse(team=team)
            tournament, snapshot_date = latest_row
            rows = list(
                session.scalars(
                    select(RawSquad).where(
                        RawSquad.team == team,
                        RawSquad.tournament == tournament,
                        RawSquad.snapshot_date == snapshot_date,
                    )
                )
            )
    except Exception as exc:
        logger.debug("team_squad: DB error for %s", team, exc_info=True)
        raise HTTPException(
            status_code=503, detail=f"team_squad DB query failed: {exc.__class__.__name__}"
        ) from exc
    rows.sort(key=lambda r: (r.shirt_number is None, r.shirt_number or 0, r.player_name))
    return TeamSquadResponse(
        team=team,
        tournament=tournament,
        snapshot_date=snapshot_date,
        players=[
            SquadMember(
                player_name=r.player_name,
                shirt_number=r.shirt_number,
                position=r.position,
                birth_date=r.birth_date,
                club=r.club,
                caps=r.caps,
                goals=r.goals,
            )
            for r in rows
        ],
    )


@router.get("/{team}/xg-form", response_model=TeamXgFormResponse)
def team_xg_form(team: str) -> TeamXgFormResponse:
    """Rolling xG aggregates over the last 5 and 10 matches with xG data.

    Reads from ``raw_match_xg`` filtered to ``team`` (regardless of whether
    they were home or away). Splits with no rows return ``matches=0`` and
    null aggregates so the dashboard can render a clean "no data" state.
    """
    # Pull a generous window (limit 60) so the 12-month aggregate has enough
    # to chew on for teams that play >10 matches a year (qualifiers, Nations
    # League, friendlies). The last-5 + last-10 buckets slice this list.
    try:
        eng = get_engine()
        with Session(eng, future=True) as session:
            rows = list(
                session.scalars(
                    select(RawMatchXg)
                    .where(RawMatchXg.team == team)
                    .order_by(RawMatchXg.match_date.desc())
                    .limit(60)
                )
            )
    except Exception as exc:
        logger.debug("team_xg_form: DB error for %s", team, exc_info=True)
        raise HTTPException(
            status_code=503, detail=f"team_xg_form DB query failed: {exc.__class__.__name__}"
        ) from exc

    def _aggregate(window: list[RawMatchXg]) -> XgFormSplit:
        if not window:
            return XgFormSplit(matches=0)
        xg_for = sum(float(r.xg_for) for r in window) / len(window)
        xg_against = sum(float(r.xg_against) for r in window) / len(window)
        return XgFormSplit(
            matches=len(window),
            xg_for=xg_for,
            xg_against=xg_against,
            xg_diff=xg_for - xg_against,
        )

    # 12-month window keyed off the freshest row in the dataset (not the
    # server clock) so the metric stays meaningful in test environments that
    # only have historical xG data.
    from datetime import timedelta  # noqa: PLC0415 — local to this aggregation

    last_12_months_rows: list[RawMatchXg] = []
    if rows:
        anchor = rows[0].match_date
        cutoff = anchor - timedelta(days=365)
        last_12_months_rows = [r for r in rows if r.match_date >= cutoff]

    return TeamXgFormResponse(
        team=team,
        last_5=_aggregate(rows[:5]),
        last_10=_aggregate(rows[:10]),
        last_12_months=_aggregate(last_12_months_rows),
    )


@router.get("/{team}/assets", response_model=TeamAssetsResponse)
def team_assets(team: str) -> TeamAssetsResponse:
    """Return crest / kit / stadium metadata for ``team`` from ``raw_team_assets``.

    Returns a payload with all-``null`` fields when there's no DB row (so the
    dashboard can render a fallback without 404-handling). Surfaces 503 only
    when Postgres itself is unreachable.
    """
    try:
        eng = get_engine()
        with Session(eng, future=True) as session:
            row = session.scalars(select(RawTeamAsset).where(RawTeamAsset.team == team)).first()
    except Exception as exc:
        logger.debug("team_assets: DB unreachable for %s", team, exc_info=True)
        raise HTTPException(
            status_code=503, detail=f"team_assets DB query failed: {exc.__class__.__name__}"
        ) from exc
    if row is None:
        return TeamAssetsResponse(team=team)
    return TeamAssetsResponse(
        team=team,
        crest_url=row.crest_url,
        kit_home_color=row.kit_home_color,
        kit_away_color=row.kit_away_color,
        stadium_name=row.stadium_name,
        stadium_capacity=row.stadium_capacity,
        stadium_city=row.stadium_city,
        stadium_country=row.stadium_country,
    )


# --- /api/v1/teams/{team}/path-to-final ------------------------------------
#
# Path-to-final caches a single Monte Carlo opponent-histogram pass at module
# scope and serves every team from it. Burning ~1-2 s once per cache window
# beats running per-team mini-sims on every Team Profile click.

PATH_CACHE_TTL_SECONDS = 3600  # mirror the standings cache window
DEFAULT_PATH_N_SIMS = 2000
DEFAULT_PATH_SEED = 42

_PATH_CACHE: dict[tuple[int, int], tuple[float, dict[str, dict[str, dict[str, int]]]]] = {}
_PATH_CACHE_LOCK = threading.Lock()


def _cached_path_histograms(
    fixtures: WC2026Fixtures,
    model: PoissonDC,
    *,
    n_sims: int,
    seed: int,
    shootout_strategy,
) -> dict[str, dict[str, dict[str, int]]]:
    key = (n_sims, seed)
    now = time.monotonic()
    with _PATH_CACHE_LOCK:
        cached = _PATH_CACHE.get(key)
        if cached is not None and now - cached[0] < PATH_CACHE_TTL_SECONDS:
            return cached[1]
    hist = compute_path_to_final(
        fixtures, model, n_sims=n_sims, seed=seed, shootout_strategy=shootout_strategy
    )
    with _PATH_CACHE_LOCK:
        _PATH_CACHE[key] = (now, hist)
    return hist


@router.get("/{team}/path-to-final", response_model=TeamPathToFinal)
def team_path_to_final(
    request: Request,
    team: str,
    n_sims: int = Query(default=DEFAULT_PATH_N_SIMS, ge=200, le=10_000),
    seed: int = Query(default=DEFAULT_PATH_SEED),
    fixtures: WC2026Fixtures = Depends(get_fixtures),
    model: PoissonDC = Depends(get_model),
) -> TeamPathToFinal:
    """Round-by-round advancement probabilities + most-likely opponents.

    Each round entry carries P(team reaches it) and a histogram of the
    opponents the team faced when they did. The default ``n_sims=2000`` is
    a cost/precision compromise — for chart-level resolution it's plenty;
    for tail-team estimates bump it to 5–10k.
    """
    shootout_strategy = getattr(request.app.state, "shootout_strategy", None)
    histograms = _cached_path_histograms(
        fixtures, model, n_sims=n_sims, seed=seed, shootout_strategy=shootout_strategy
    )
    team_hist = histograms.get(team)
    if team_hist is None:
        # Team isn't in the WC 2026 corpus — surface a clean 422 rather than
        # an empty-everywhere payload that looks like "we ran 0 sims".
        raise HTTPException(status_code=422, detail=f"unknown team: {team!r}")
    rounds: list[PathRound] = []
    for label in PATH_ROUND_MATCH_IDS:
        counts = team_hist.get(label, {})
        total = sum(counts.values())
        p_reached = total / n_sims if n_sims > 0 else 0.0
        opponents = [
            PathOpponent(team=opp, p_conditional=c / total)
            for opp, c in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        ]
        rounds.append(
            PathRound(
                round=label,
                p_reached=p_reached,
                most_likely_opponent=opponents[0] if opponents else None,
                top_opponents=opponents[:3],
            )
        )
    return TeamPathToFinal(team=team, n_sims=n_sims, rounds=rounds)
