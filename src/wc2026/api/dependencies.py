"""FastAPI dependencies — pull the cached model and fixtures off app.state."""

from __future__ import annotations

from fastapi import HTTPException, Request, status

from wc2026.models.poisson_dc import PoissonDC
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
