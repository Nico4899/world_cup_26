"""Fixture-list routes."""

from __future__ import annotations

from datetime import date as date_type

from fastapi import APIRouter, Depends, HTTPException, Query, status

from wc2026.api.dependencies import get_fixtures, get_model
from wc2026.api.routes.predictions import build_prediction
from wc2026.api.schemas import FixtureSummary, FixtureWithPrediction
from wc2026.models.poisson_dc import PoissonDC
from wc2026.sim.fixtures import WC2026Fixtures

router = APIRouter(prefix="/api/v1/matches")


def _fixture_summary(idx: int, fixtures: WC2026Fixtures) -> FixtureSummary:
    m = fixtures.matches[idx]
    # FixtureMatch.date is always a pd.Timestamp per the dataclass; .date() yields a date.
    return FixtureSummary(
        match_id=idx,
        date=m.date.date(),
        home_team=m.home_team,
        away_team=m.away_team,
        group=m.group,
        city=m.city,
        country=m.country,
        neutral=m.neutral,
    )


@router.get("", response_model=list[FixtureSummary])
def list_matches(
    date: date_type | None = Query(default=None, description="ISO date filter"),
    group: str | None = Query(default=None, description="Group letter A..L"),
    fixtures: WC2026Fixtures = Depends(get_fixtures),
) -> list[FixtureSummary]:
    out: list[FixtureSummary] = []
    for idx, m in enumerate(fixtures.matches):
        if date is not None and m.date.date() != date:
            continue
        if group is not None and m.group != group.upper():
            continue
        out.append(_fixture_summary(idx, fixtures))
    return out


@router.get("/{match_id}", response_model=FixtureWithPrediction)
def get_match(
    match_id: int,
    fixtures: WC2026Fixtures = Depends(get_fixtures),
    model: PoissonDC = Depends(get_model),
) -> FixtureWithPrediction:
    if not 0 <= match_id < len(fixtures.matches):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"match_id {match_id} out of range [0, {len(fixtures.matches)})",
        )
    summary = _fixture_summary(match_id, fixtures)
    fx = fixtures.matches[match_id]
    # Include the full score matrix so Match Detail can render the heatmap in one call.
    pred = build_prediction(
        model,
        fx.home_team,
        fx.away_team,
        neutral=fx.neutral,
        top_n=5,
        include_matrix=True,
    )
    return FixtureWithPrediction(fixture=summary, prediction=pred)
