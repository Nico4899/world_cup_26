"""Health check route."""

from __future__ import annotations

from datetime import UTC, date, datetime

from fastapi import APIRouter, Request

from wc2026.api.schemas import HealthResponse

router = APIRouter()


def _age_days(snapshot_date: date | None) -> int | None:
    if snapshot_date is None:
        return None
    today = datetime.now(UTC).date()
    return max(0, (today - snapshot_date).days)


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    state = request.app.state
    model = getattr(state, "model", None)
    fitted = bool(model is not None and model.fitted)
    n_teams = len(model.params_.teams) if fitted else 0
    snapshot_date = getattr(state, "elo_snapshot_date", None)
    return HealthResponse(
        status="ok",
        model_fitted=fitted,
        model_teams_n=n_teams,
        model_fit_at=getattr(state, "model_fit_at", None),
        model_version=getattr(state, "model_version", None),
        elo_snapshot_date=snapshot_date,
        elo_snapshot_age_days=_age_days(snapshot_date),
        shootout_model_loaded=getattr(state, "shootout_model", None) is not None,
    )
