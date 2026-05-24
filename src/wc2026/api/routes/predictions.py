"""Pairwise-prediction routes."""

from __future__ import annotations

from datetime import date as date_cls
from datetime import datetime

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from wc2026.api.dependencies import get_feature_sources, get_model, get_xgb_model
from wc2026.api.schemas import (
    BlendedOutcome,
    OutcomeProbabilities,
    PredictionResponse,
    Scoreline,
)
from wc2026.features.build_match_features import (
    FeatureSources,
    MatchSpec,
    build_features_for_match,
)
from wc2026.models.blend import blend_dict
from wc2026.models.poisson_dc import PoissonDC
from wc2026.models.xgb_classifier import (
    CLASS_AWAY,
    CLASS_DRAW,
    CLASS_HOME,
    DEFAULT_FEATURE_COLUMNS,
    XgbMatchModel,
)

router = APIRouter(prefix="/api/v1/predictions")

DEFAULT_TOP_SCORELINES = 5
STATUS_UNPROCESSABLE = 422  # Starlette deprecated the named constant
DEFAULT_BLEND_WEIGHT = 0.5


def _xgb_outcome(
    xgb_model: XgbMatchModel,
    sources: FeatureSources,
    *,
    home: str,
    away: str,
    neutral: bool,
    match_date: date_cls | None = None,
) -> dict[str, float] | None:
    """Build features for one matchup and return the XGB 1X2 dict.

    Returns ``None`` when the feature build raises — the prediction route then
    silently drops the blend, keeping the Poisson-only outcome intact.
    """
    spec = MatchSpec(
        match_date=match_date or datetime.now().date(),
        home_team=home,
        away_team=away,
        neutral=neutral,
    )
    try:
        feature_dict = build_features_for_match(spec, sources)
    except (KeyError, ValueError):
        return None
    feature_row = {k: feature_dict.get(k) for k in DEFAULT_FEATURE_COLUMNS}
    import pandas as pd  # noqa: PLC0415

    probs = xgb_model.predict_proba(pd.DataFrame([feature_row]))[0]
    return {
        "home_win": float(probs[CLASS_HOME]),
        "draw": float(probs[CLASS_DRAW]),
        "away_win": float(probs[CLASS_AWAY]),
    }


def build_prediction(
    model: PoissonDC,
    home: str,
    away: str,
    *,
    neutral: bool,
    top_n: int = DEFAULT_TOP_SCORELINES,
    include_matrix: bool = False,
    xgb_model: XgbMatchModel | None = None,
    feature_sources: FeatureSources | None = None,
    blend: bool = False,
    blend_weight: float = DEFAULT_BLEND_WEIGHT,
    match_date: date_cls | None = None,
) -> PredictionResponse:
    """Compute a full prediction payload for (home, away). Raises 422 on unknown team.

    The full ``score_matrix`` (~121 floats) is only included when
    ``include_matrix=True`` to keep list-endpoint payloads small.
    """
    try:
        lh, la = model.expected_goals(home, away, neutral=neutral)
        outcome = model.outcome_probs(home, away, neutral=neutral)
        score_matrix = model.score_probs(home, away, neutral=neutral)
    except KeyError as exc:
        # PoissonDC already raises KeyError("unknown team: <name>"); surface as-is
        # to avoid duplicating the "unknown team:" prefix in the response detail.
        detail = exc.args[0] if exc.args else "unknown team"
        raise HTTPException(status_code=STATUS_UNPROCESSABLE, detail=detail) from exc

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
    poisson_outcome = OutcomeProbabilities(
        home_win=float(outcome["home_win"]),
        draw=float(outcome["draw"]),
        away_win=float(outcome["away_win"]),
    )
    response_outcome = poisson_outcome
    blend_payload: BlendedOutcome | None = None
    if blend and xgb_model is not None and feature_sources is not None:
        xgb_outcome_dict = _xgb_outcome(
            xgb_model,
            feature_sources,
            home=home,
            away=away,
            neutral=neutral,
            match_date=match_date,
        )
        if xgb_outcome_dict is not None:
            poisson_dict = {
                "home_win": poisson_outcome.home_win,
                "draw": poisson_outcome.draw,
                "away_win": poisson_outcome.away_win,
            }
            blended_dict = blend_dict(poisson_dict, xgb_outcome_dict, weight=blend_weight)
            blend_payload = BlendedOutcome(
                poisson=poisson_outcome,
                xgb=OutcomeProbabilities(**xgb_outcome_dict),
                blended=OutcomeProbabilities(**blended_dict),
                weight=blend_weight,
            )
            response_outcome = OutcomeProbabilities(**blended_dict)
    return PredictionResponse(
        home_team=home,
        away_team=away,
        neutral=neutral,
        expected_home_goals=float(lh),
        expected_away_goals=float(la),
        outcome=response_outcome,
        top_scorelines=top,
        score_matrix=score_matrix.tolist() if include_matrix else None,
        blend=blend_payload,
    )


@router.get("/{home}/{away}", response_model=PredictionResponse)
def pairwise_prediction(
    request: Request,
    home: str,
    away: str,
    neutral: bool = Query(default=True, description="Treat as neutral venue"),
    blend: bool = Query(
        default=False,
        description="Blend with the XGB classifier when available. No-op when XGB isn't loaded.",
    ),
    blend_weight: float = Query(
        default=DEFAULT_BLEND_WEIGHT,
        ge=0.0,
        le=1.0,
        description="Poisson mixing weight in [0, 1]; XGB gets 1 - weight.",
    ),
    model: PoissonDC = Depends(get_model),
) -> PredictionResponse:
    xgb_model = get_xgb_model(request)
    feature_sources = get_feature_sources(request)
    return build_prediction(
        model,
        home,
        away,
        neutral=neutral,
        include_matrix=True,
        xgb_model=xgb_model,
        feature_sources=feature_sources,
        blend=blend,
        blend_weight=blend_weight,
    )
