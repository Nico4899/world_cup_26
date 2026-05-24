"""SHAP-based per-match explanation route.

``GET /api/v1/explain/{match_id}`` returns the top-N feature contributions
toward the predicted class (default: home_win) for one WC 2026 fixture. The
endpoint requires the optional XGB+SHAP artefacts to be loaded; without them
it returns 503 so the dashboard can degrade gracefully.
"""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from wc2026.api.dependencies import (
    get_feature_sources,
    get_fixtures,
    get_model,
    require_xgb_explainer,
)
from wc2026.api.schemas import (
    FeatureContributionItem,
    MatchExplanation,
    OutcomeProbabilities,
)
from wc2026.features.build_match_features import (
    FeatureSources,
    MatchSpec,
    build_features_for_match,
)
from wc2026.models.poisson_dc import PoissonDC
from wc2026.models.shap_explain import CLASS_NAMES, XgbExplainer
from wc2026.models.xgb_classifier import (
    CLASS_AWAY,
    CLASS_DRAW,
    CLASS_HOME,
    DEFAULT_FEATURE_COLUMNS,
)
from wc2026.sim.fixtures import WC2026Fixtures

router = APIRouter(prefix="/api/v1/explain")

DEFAULT_TOP_N = 5

_CLASS_BY_NAME: dict[str, int] = {name: idx for idx, name in CLASS_NAMES.items()}


def _resolve_class_index(class_name: str) -> int:
    if class_name not in _CLASS_BY_NAME:
        raise HTTPException(
            status_code=422,
            detail=f"class must be one of {sorted(_CLASS_BY_NAME)}; got {class_name!r}",
        )
    return _CLASS_BY_NAME[class_name]


def _resolve_match(fixtures: WC2026Fixtures, match_id: int):
    if match_id < 0 or match_id >= len(fixtures.matches):
        raise HTTPException(
            status_code=404,
            detail=f"match_id {match_id} out of range (0..{len(fixtures.matches) - 1})",
        )
    return fixtures.matches[match_id]


@router.get("/{match_id}", response_model=MatchExplanation)
def explain_match(
    request: Request,
    match_id: int,
    class_name: str = Query(
        default="home_win",
        description=f"Which outcome to explain — one of {sorted(_CLASS_BY_NAME)}.",
    ),
    top_n: int = Query(
        default=DEFAULT_TOP_N,
        ge=1,
        le=len(DEFAULT_FEATURE_COLUMNS),
        description="How many top features to include in the contributions list.",
    ),
    explainer: XgbExplainer = Depends(require_xgb_explainer),
    model: PoissonDC = Depends(get_model),
    fixtures: WC2026Fixtures = Depends(get_fixtures),
    sources: FeatureSources = Depends(get_feature_sources),
) -> MatchExplanation:
    class_index = _resolve_class_index(class_name)
    match = _resolve_match(fixtures, match_id)
    match_date = match.date.date() if hasattr(match.date, "date") else match.date
    spec = MatchSpec(
        match_date=match_date,
        home_team=match.home_team,
        away_team=match.away_team,
        neutral=match.neutral,
    )
    features = build_features_for_match(spec, sources)
    feature_row = {k: features.get(k) for k in DEFAULT_FEATURE_COLUMNS}
    contributions = explainer.top_features(
        pd.DataFrame([feature_row]), class_index=class_index, n=top_n
    )
    items = [
        FeatureContributionItem(feature=c.feature, value=c.value, contribution=c.contribution)
        for c in contributions
    ]
    poisson_outcome_dict = model.outcome_probs(
        match.home_team, match.away_team, neutral=match.neutral
    )
    poisson_outcome = OutcomeProbabilities(
        home_win=float(poisson_outcome_dict["home_win"]),
        draw=float(poisson_outcome_dict["draw"]),
        away_win=float(poisson_outcome_dict["away_win"]),
    )
    xgb_outcome: OutcomeProbabilities | None = None
    xgb_model = getattr(request.app.state, "xgb_model", None)
    if xgb_model is not None:
        probs = xgb_model.predict_proba(pd.DataFrame([feature_row]))[0]
        xgb_outcome = OutcomeProbabilities(
            home_win=float(probs[CLASS_HOME]),
            draw=float(probs[CLASS_DRAW]),
            away_win=float(probs[CLASS_AWAY]),
        )
    return MatchExplanation(
        home_team=match.home_team,
        away_team=match.away_team,
        match_date=match_date,
        class_name=class_name,
        contributions=items,
        poisson_outcome=poisson_outcome,
        xgb_outcome=xgb_outcome,
    )
