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

from wc2026.api.routes import (
    explain,
    h2h,
    health,
    live,
    matches,
    ops,
    predictions,
    teams,
    tournament,
    track_record,
)
from wc2026.features.build_match_features import FeatureSources
from wc2026.features.match_weights import combined_weight
from wc2026.ingest.eloratings_scraper import load_latest_snapshot
from wc2026.ingest.kaggle_intl import load_played
from wc2026.models.live_win_prob import (
    DEFAULT_ARTIFACT_PATH as LIVE_WIN_PROB_PATH,
)
from wc2026.models.live_win_prob import (
    LiveWinProbModel,
)
from wc2026.models.poisson_dc import PoissonDC, hydrate_from_artefact
from wc2026.models.shap_explain import XgbExplainer
from wc2026.models.shootout import (
    fit_shootout_model,
    load_historical_shootouts,
    load_shootout_model,
    simulate_shootout,
)
from wc2026.models.xgb_classifier import (
    DEFAULT_META_PATH as XGB_DEFAULT_META_PATH,
)
from wc2026.models.xgb_classifier import (
    DEFAULT_MODEL_PATH as XGB_DEFAULT_MODEL_PATH,
)
from wc2026.models.xgb_classifier import (
    XgbMatchModel,
)
from wc2026.observability.sentry import init_sentry
from wc2026.sim.fixtures import load_wc2026_fixtures

ARTEFACT_PATH = Path("data/artifacts/poisson_dc/latest.npz")
SHOOTOUT_ARTEFACT_PATH = Path("data/artifacts/shootout/latest.json")
XGB_MODEL_PATH = XGB_DEFAULT_MODEL_PATH
XGB_META_PATH = XGB_DEFAULT_META_PATH

# Match Stage 0.6's tuned defaults.
MODEL_HALF_LIFE_DAYS = 3650.0
MODEL_HISTORY_YEARS = 10
MODEL_VERSION = "poisson_dc.v1"


def _today_utc_ts() -> pd.Timestamp:
    """Reference date for the cold-start lifespan fit.

    Kept as a function (not a module constant) so each `uvicorn` boot uses the
    current day, not a date frozen at import time. Tests can monkey-patch this.
    """
    return pd.Timestamp(datetime.now(UTC).date())


CORS_ORIGINS = (
    "http://localhost:8501",
    "http://localhost:3000",
    "http://127.0.0.1:8501",
    "http://127.0.0.1:3000",
)


def _fit_model(df: pd.DataFrame) -> PoissonDC:
    ref_date = _today_utc_ts()
    cutoff = ref_date - pd.Timedelta(days=int(MODEL_HISTORY_YEARS * 365.25))
    train = df[df["date"] >= cutoff].reset_index(drop=True)
    weights = combined_weight(train, ref_date=ref_date, half_life_days=MODEL_HALF_LIFE_DAYS)
    return PoissonDC().fit(train, weights=weights)


def _load_or_fit_model(df: pd.DataFrame) -> tuple[PoissonDC, str, datetime]:
    """Prefer a freshly-fit artefact on disk over a cold lifespan-time fit.

    Returns ``(model, source, fit_at)`` where source is ``"artefact"`` (and
    ``fit_at`` is the artefact mtime) or ``"in_process_fit"`` (and ``fit_at``
    is now).
    """
    if ARTEFACT_PATH.exists():
        try:
            model = hydrate_from_artefact(ARTEFACT_PATH)
        except (OSError, ValueError, KeyError):
            # Corrupt or schema-mismatched artefact: silently re-fit from scratch.
            pass
        else:
            mtime = datetime.fromtimestamp(ARTEFACT_PATH.stat().st_mtime, tz=UTC)
            return model, "artefact", mtime
    return _fit_model(df), "in_process_fit", datetime.now(UTC)


def _load_live_win_prob_model() -> LiveWinProbModel | None:
    """Optional load of the Phase 6 live in-running win-prob model.

    Returns ``None`` when no artefact is on disk — the /live endpoints then
    fall back to the pre-match Poisson probability.
    """
    if not LIVE_WIN_PROB_PATH.exists():
        return None
    try:
        return LiveWinProbModel.load(LIVE_WIN_PROB_PATH)
    except (OSError, ValueError, KeyError):
        return None


def _load_xgb_model() -> XgbMatchModel | None:
    """Best-effort load of the Phase 5 XGB H/D/A classifier.

    Returns ``None`` if either the model JSON or its sidecar meta file is
    missing — callers degrade gracefully to Poisson-only predictions.
    """
    if not (XGB_MODEL_PATH.exists() and XGB_META_PATH.exists()):
        return None
    try:
        return XgbMatchModel.load(XGB_MODEL_PATH, XGB_META_PATH)
    except (OSError, ValueError, KeyError):
        return None


def _build_xgb_feature_sources(
    played_df: pd.DataFrame, model: PoissonDC, elo_snapshot
) -> FeatureSources:
    """Cache the inputs the /predictions + /explain routes feed into the feature builder.

    We populate ``elo_by_team`` from the snapshot already loaded for the
    shootout submodel and pass the played-matches df straight through for the
    rest-days feature. xG history and squad ages are left ``None`` until the
    relevant ingesters have populated their parquet snapshots — XGBoost
    handles the resulting NaNs natively.
    """
    elo_by_team: dict[str, float] | None = None
    if elo_snapshot is not None and "team_name" in elo_snapshot.columns:
        elo_by_team = {
            str(name): float(rating)
            for name, rating in zip(elo_snapshot["team_name"], elo_snapshot["rating"], strict=True)
            if pd.notna(name) and pd.notna(rating)
        }
    return FeatureSources(
        elo_by_team=elo_by_team,
        matches=played_df,
        poisson_model=model,
    )


def _build_shootout_strategy():
    """Resolve the Elo-based shootout submodel; return a strategy + metadata.

    Resolution order:
        1. ``data/artifacts/shootout/latest.json`` — fresh from the daily refit.
        2. In-process fit against the on-disk Elo snapshot + shootouts CSV.
        3. None (knockout simulator's 50/50 coin flip takes over).

    Returns ``(strategy, fitted_model, elo_snapshot_date)``. The snapshot date is
    surfaced on /health so operators can spot a stale Elo file.
    """
    model = None
    if SHOOTOUT_ARTEFACT_PATH.exists():
        try:
            model = load_shootout_model(SHOOTOUT_ARTEFACT_PATH)
        except (OSError, ValueError, KeyError):
            model = None

    elo = None
    if model is None:
        try:
            elo = load_latest_snapshot()
            shootouts = load_historical_shootouts()
            model = fit_shootout_model(shootouts, elo)
        except (FileNotFoundError, ValueError):
            return None, None, None
    else:
        # Still load the snapshot to surface its freshness on /health, but don't
        # let a snapshot-read failure invalidate an already-loaded artefact.
        try:
            elo = load_latest_snapshot()
        except (FileNotFoundError, ValueError):
            elo = None

    def strategy(home: str, away: str, rng) -> str:
        return simulate_shootout(home, away, model, None, rng)

    snapshot_date = None
    if elo is not None and "snapshot_date" in elo.columns and not elo["snapshot_date"].empty:
        snapshot_date = pd.to_datetime(elo["snapshot_date"].iloc[0]).date()
    return strategy, model, snapshot_date


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Sentry init runs first so any subsequent lifespan crash is reported.
    # No-op when SENTRY_DSN is unset.
    init_sentry(service="api")
    # Cache the full played-matches DataFrame once: the model fit uses a
    # 10-year window of it, and the /teams/{t}/recent and /h2h endpoints
    # query against it directly.
    app.state.played_df = load_played()
    model, source, fit_at = _load_or_fit_model(app.state.played_df)
    app.state.model = model
    app.state.model_source = source
    app.state.model_fit_at = fit_at
    app.state.fixtures = load_wc2026_fixtures()
    app.state.model_version = MODEL_VERSION
    # Optional Elo-based shootout model; None falls back to the 50/50 placeholder.
    (
        app.state.shootout_strategy,
        app.state.shootout_model,
        app.state.elo_snapshot_date,
    ) = _build_shootout_strategy()
    # Optional Phase 5 XGB classifier + SHAP explainer; both None when no
    # artefact has been produced yet — predictions then stay Poisson-only.
    xgb_model = _load_xgb_model()
    app.state.xgb_model = xgb_model
    app.state.xgb_explainer = XgbExplainer.from_model(xgb_model) if xgb_model is not None else None
    # Capture an Elo snapshot once for the feature-builder path (the shootout
    # branch already loaded it, but its reference isn't otherwise retained).
    try:
        elo_snapshot = load_latest_snapshot()
    except (FileNotFoundError, ValueError):
        elo_snapshot = None
    app.state.feature_sources = _build_xgb_feature_sources(app.state.played_df, model, elo_snapshot)
    # Optional Phase 6 live in-running win-prob model.
    app.state.live_win_prob_model = _load_live_win_prob_model()
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
app.include_router(ops.router)
app.include_router(explain.router)
app.include_router(live.router)
app.include_router(track_record.router)
