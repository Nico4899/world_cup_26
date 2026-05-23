"""FastAPI app entrypoint for the WC 2026 prediction platform.

Run locally with:
    uv run uvicorn wc2026.api.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from wc2026.api.routes import health, matches, predictions, tournament
from wc2026.features.match_weights import combined_weight
from wc2026.ingest.kaggle_intl import load_played, load_scheduled
from wc2026.models.poisson_dc import PoissonDC
from wc2026.sim.fixtures import parse_wc2026_fixtures

# Match Stage 0.6's tuned defaults.
MODEL_HALF_LIFE_DAYS = 3650.0
MODEL_HISTORY_YEARS = 10
MODEL_REF_DATE = pd.Timestamp("2026-05-23")
MODEL_VERSION = "poisson_dc.v1"

CORS_ORIGINS = (
    "http://localhost:8501",
    "http://localhost:3000",
    "http://127.0.0.1:8501",
    "http://127.0.0.1:3000",
)


def _fit_model() -> PoissonDC:
    df = load_played()
    cutoff = MODEL_REF_DATE - pd.Timedelta(days=int(MODEL_HISTORY_YEARS * 365.25))
    df = df[df["date"] >= cutoff].reset_index(drop=True)
    weights = combined_weight(df, ref_date=MODEL_REF_DATE, half_life_days=MODEL_HALF_LIFE_DAYS)
    return PoissonDC().fit(df, weights=weights)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = _fit_model()
    app.state.fixtures = parse_wc2026_fixtures(load_scheduled())
    app.state.model_version = MODEL_VERSION
    app.state.model_fit_at = datetime.now(UTC)
    yield


app = FastAPI(
    title="WC 2026 Predictions API",
    version="0.1.0",
    description="Calibrated probabilistic predictions for FIFA World Cup 2026.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(CORS_ORIGINS),
    allow_methods=["GET"],
    allow_headers=["*"],
    allow_credentials=False,
)

app.include_router(health.router)
app.include_router(matches.router)
app.include_router(predictions.router)
app.include_router(tournament.router)
