"""Per-team queries — recent form from the team's perspective."""

from __future__ import annotations

import logging

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from wc2026.api.dependencies import get_played_df
from wc2026.api.schemas import (
    EloHistoryPoint,
    FifaRankingPoint,
    SquadMember,
    TeamAssetsResponse,
    TeamEloHistory,
    TeamFifaRankingHistory,
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
    try:
        eng = get_engine()
        with Session(eng, future=True) as session:
            rows = list(
                session.scalars(
                    select(RawMatchXg)
                    .where(RawMatchXg.team == team)
                    .order_by(RawMatchXg.match_date.desc())
                    .limit(10)
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

    return TeamXgFormResponse(
        team=team,
        last_5=_aggregate(rows[:5]),
        last_10=_aggregate(rows),
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
