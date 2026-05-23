"""Head-to-head queries between two teams."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from wc2026.api.dependencies import get_played_df
from wc2026.api.schemas import H2HMatch

router = APIRouter(prefix="/api/v1/h2h")

DEFAULT_H2H_N = 10
MAX_H2H_N = 50


@router.get("/{team_a}/{team_b}", response_model=list[H2HMatch])
def head_to_head(
    team_a: str,
    team_b: str,
    n: int = Query(default=DEFAULT_H2H_N, ge=1, le=MAX_H2H_N),
    played: pd.DataFrame = Depends(get_played_df),
) -> list[H2HMatch]:
    """Return the ``n`` most recent matches between ``team_a`` and ``team_b``.

    Returns ``[]`` if the two teams have never met — that's a valid answer, not
    a 422. A 422 is raised only if one of the team names is entirely unknown
    in the played-matches dataset (likely a typo).
    """
    pair_mask = ((played["home_team"] == team_a) & (played["away_team"] == team_b)) | (
        (played["home_team"] == team_b) & (played["away_team"] == team_a)
    )
    sub = played[pair_mask].sort_values("date", ascending=False).head(n)
    if sub.empty:
        # Distinguish "never met" from "neither team exists".
        for team in (team_a, team_b):
            if not (played["home_team"].eq(team) | played["away_team"].eq(team)).any():
                raise HTTPException(status_code=422, detail=f"unknown team: {team!r}")
        return []

    return [
        H2HMatch(
            date=row["date"].date(),
            home_team=row["home_team"],
            away_team=row["away_team"],
            home_score=int(row["home_score"]),
            away_score=int(row["away_score"]),
            tournament=str(row["tournament"]),
            neutral=bool(row["neutral"]),
        )
        for _, row in sub.iterrows()
    ]
