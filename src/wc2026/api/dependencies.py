"""FastAPI dependencies — pull the cached model and fixtures off app.state."""

from __future__ import annotations

import pandas as pd
from fastapi import HTTPException, Request, status

from wc2026.features.build_match_features import FeatureSources
from wc2026.models.poisson_dc import PoissonDC
from wc2026.models.shap_explain import XgbExplainer
from wc2026.models.xgb_classifier import XgbMatchModel
from wc2026.sim.fixtures import WC2026Fixtures


def get_model(request: Request) -> PoissonDC:
    model = getattr(request.app.state, "model", None)
    if model is None or not model.fitted:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model not loaded",
        )
    return model


def get_fixtures(request: Request) -> WC2026Fixtures:
    fixtures = getattr(request.app.state, "fixtures", None)
    if fixtures is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="fixtures not loaded",
        )
    return fixtures


def get_played_df(request: Request) -> pd.DataFrame:
    df = getattr(request.app.state, "played_df", None)
    if df is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="played-matches dataset not loaded",
        )
    return df


def get_xgb_model(request: Request) -> XgbMatchModel | None:
    """Optional dependency: returns ``None`` if no XGB artefact was loaded.

    Routes that need XGB should branch on the return value rather than 503ing,
    so the API degrades gracefully when the Phase 5 model is unavailable.
    """
    return getattr(request.app.state, "xgb_model", None)


def get_xgb_explainer(request: Request) -> XgbExplainer | None:
    """Optional dependency: SHAP explainer wrapped around the loaded XGB."""
    return getattr(request.app.state, "xgb_explainer", None)


def require_xgb_explainer(request: Request) -> XgbExplainer:
    """503 if no explainer is loaded — used by the /explain route."""
    explainer = get_xgb_explainer(request)
    if explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "XGB explainer not loaded — refit via scripts/refit_xgb.py and restart the API."
            ),
        )
    return explainer


def get_feature_sources(request: Request) -> FeatureSources:
    """Return a (cheaply cloned) bundle of the feature sources cached at lifespan."""
    bundle = getattr(request.app.state, "feature_sources", None)
    if bundle is None:
        return FeatureSources()
    return bundle
