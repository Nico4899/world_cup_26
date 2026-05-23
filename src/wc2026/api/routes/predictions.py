"""Pairwise-prediction routes."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query

from wc2026.api.dependencies import get_model
from wc2026.api.schemas import (
    OutcomeProbabilities,
    PredictionResponse,
    Scoreline,
)
from wc2026.models.poisson_dc import PoissonDC

router = APIRouter(prefix="/api/v1/predictions")

DEFAULT_TOP_SCORELINES = 5
STATUS_UNPROCESSABLE = 422  # Starlette deprecated the named constant


def build_prediction(
    model: PoissonDC,
    home: str,
    away: str,
    *,
    neutral: bool,
    top_n: int = DEFAULT_TOP_SCORELINES,
) -> PredictionResponse:
    """Compute a full prediction payload for (home, away). Raises 422 on unknown team."""
    try:
        lh, la = model.expected_goals(home, away, neutral=neutral)
        outcome = model.outcome_probs(home, away, neutral=neutral)
        score_matrix = model.score_probs(home, away, neutral=neutral)
    except KeyError as exc:
        raise HTTPException(
            status_code=STATUS_UNPROCESSABLE,
            detail=f"unknown team: {exc.args[0] if exc.args else exc!s}",
        ) from exc

    flat = score_matrix.ravel()
    cap = min(top_n, flat.size)
    top_idx = np.argpartition(flat, -cap)[-cap:]
    top_idx = top_idx[np.argsort(flat[top_idx])[::-1]]
    m = score_matrix.shape[0]
    top = [
        Scoreline(
            home_goals=int(idx // m),
            away_goals=int(idx % m),
            probability=float(flat[idx]),
        )
        for idx in top_idx
    ]
    return PredictionResponse(
        home_team=home,
        away_team=away,
        neutral=neutral,
        expected_home_goals=float(lh),
        expected_away_goals=float(la),
        outcome=OutcomeProbabilities(
            home_win=float(outcome["home_win"]),
            draw=float(outcome["draw"]),
            away_win=float(outcome["away_win"]),
        ),
        top_scorelines=top,
    )


@router.get("/{home}/{away}", response_model=PredictionResponse)
def pairwise_prediction(
    home: str,
    away: str,
    neutral: bool = Query(default=True, description="Treat as neutral venue"),
    model: PoissonDC = Depends(get_model),
) -> PredictionResponse:
    return build_prediction(model, home, away, neutral=neutral)
