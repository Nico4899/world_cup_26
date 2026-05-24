"""Per-team queries — recent form from the team's perspective."""

from __future__ import annotations

import logging

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from wc2026.api.dependencies import get_played_df
from wc2026.api.schemas import TeamAssetsResponse, TeamRecentMatch
from wc2026.db.models import RawTeamAsset
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
            row = session.scalars(
                select(RawTeamAsset).where(RawTeamAsset.team == team)
            ).first()
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
