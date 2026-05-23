"""Health check route."""

from __future__ import annotations

from fastapi import APIRouter, Request

from wc2026.api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    state = request.app.state
    model = getattr(state, "model", None)
    fitted = bool(model is not None and model.fitted)
    n_teams = len(model.params_.teams) if fitted else 0
    return HealthResponse(
        status="ok",
        model_fitted=fitted,
        model_teams_n=n_teams,
        model_fit_at=getattr(state, "model_fit_at", None),
        model_version=getattr(state, "model_version", None),
    )
