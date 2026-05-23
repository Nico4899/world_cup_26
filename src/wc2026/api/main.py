"""FastAPI app entrypoint for the WC 2026 prediction platform.

Run locally with:
    uv run uvicorn wc2026.api.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from wc2026.api.routes import h2h, health, matches, predictions, teams, tournament
from wc2026.features.match_weights import combined_weight
from wc2026.ingest.eloratings_scraper import load_latest_snapshot
from wc2026.ingest.kaggle_intl import load_played, load_scheduled
from wc2026.models.poisson_dc import PoissonDC, PoissonDCParams
from wc2026.models.shootout import (
    fit_shootout_model,
    load_historical_shootouts,
    simulate_shootout,
)
from wc2026.sim.fixtures import load_group_assignment, parse_wc2026_fixtures

ARTEFACT_PATH = Path("data/artifacts/poisson_dc/latest.npz")
GROUP_ASSIGNMENT_PATH = Path("data/wc2026_group_assignment.json")

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


def _fit_model(df: pd.DataFrame) -> PoissonDC:
    cutoff = MODEL_REF_DATE - pd.Timedelta(days=int(MODEL_HISTORY_YEARS * 365.25))
    train = df[df["date"] >= cutoff].reset_index(drop=True)
    weights = combined_weight(train, ref_date=MODEL_REF_DATE, half_life_days=MODEL_HALF_LIFE_DAYS)
    return PoissonDC().fit(train, weights=weights)


def _load_or_fit_model(df: pd.DataFrame) -> tuple[PoissonDC, str, datetime]:
    """Prefer a freshly-fit artefact on disk over a cold lifespan-time fit.

    Returns ``(model, source, fit_at)`` where source is ``"artefact"`` (and
    ``fit_at`` is the artefact mtime) or ``"in_process_fit"`` (and ``fit_at``
    is now).
    """
    if ARTEFACT_PATH.exists():
        try:
            params = PoissonDCParams.load(ARTEFACT_PATH)
        except (OSError, ValueError, KeyError):
            # Corrupt or schema-mismatched artefact: silently re-fit from scratch.
            pass
        else:
            model = PoissonDC()
            model.params_ = params
            model._team_idx = {t: i for i, t in enumerate(params.teams)}
            model.converged_ = True
            mtime = datetime.fromtimestamp(ARTEFACT_PATH.stat().st_mtime, tz=UTC)
            return model, "artefact", mtime
    return _fit_model(df), "in_process_fit", datetime.now(UTC)


def _build_shootout_strategy():
    """Try to fit the Elo-based shootout submodel; return a strategy + metadata.

    If the elo snapshot or the shootouts.csv are missing — or if the fit fails
    (too few overlapping teams) — we silently fall back to ``(None, None, None)``
    and the knockout simulator's built-in 50/50 coin flip takes over.

    Returns ``(strategy, fitted_model, elo_snapshot_date)``. The snapshot date is
    surfaced on /health so operators can spot a stale Elo file.
    """
    try:
        elo = load_latest_snapshot()
        shootouts = load_historical_shootouts()
        model = fit_shootout_model(shootouts, elo)
    except (FileNotFoundError, ValueError):
        return None, None, None

    def strategy(home: str, away: str, rng) -> str:
        return simulate_shootout(home, away, model, None, rng)

    snapshot_date = None
    if "snapshot_date" in elo.columns and not elo["snapshot_date"].empty:
        snapshot_date = pd.to_datetime(elo["snapshot_date"].iloc[0]).date()
    return strategy, model, snapshot_date


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cache the full played-matches DataFrame once: the model fit uses a
    # 10-year window of it, and the /teams/{t}/recent and /h2h endpoints
    # query against it directly.
    app.state.played_df = load_played()
    model, source, fit_at = _load_or_fit_model(app.state.played_df)
    app.state.model = model
    app.state.model_source = source
    app.state.model_fit_at = fit_at
    override = None
    if GROUP_ASSIGNMENT_PATH.exists():
        try:
            override = load_group_assignment(GROUP_ASSIGNMENT_PATH)
        except (OSError, ValueError):
            # Bad/stale JSON — fall back to derived-from-dates labelling.
            override = None
    app.state.fixtures = parse_wc2026_fixtures(load_scheduled(), override_assignment=override)
    app.state.model_version = MODEL_VERSION
    # Optional Elo-based shootout model; None falls back to the 50/50 placeholder.
    (
        app.state.shootout_strategy,
        app.state.shootout_model,
        app.state.elo_snapshot_date,
    ) = _build_shootout_strategy()
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
app.include_router(teams.router)
app.include_router(h2h.router)
