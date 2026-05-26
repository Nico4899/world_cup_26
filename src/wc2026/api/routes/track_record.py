"""WC 2026 rolling calibration + historical hindcast endpoints.

* ``GET /api/v1/track-record/wc2026`` — ``RollingCalibration`` over every
  completed WC 2026 match: aggregate log-loss / Brier / RPS + per-match
  diagnostics. Backs the dashboard's live Track Record panel.
* ``GET /api/v1/track-record/historical/{tournament}`` — headline metrics +
  reliability bins for WC 2018 / WC 2022. The frontend consumes this endpoint
  so the heavy hindcast compute (refits ~10 PoissonDC models) stays on the
  API host instead of running in the browser session.

Implementation notes
--------------------
``RawLiveEvent.match_id`` holds the football-data.org numeric id; our
predictions live under the (match_date, home_team, away_team) natural key.
We bridge them by reading the latest cached football-data.org WC fixtures
off disk — no live API call, no Postgres dependency beyond the rows the
live poller already persisted.

The historical hindcast cache is module-level with a 24 h TTL; the actual
``hindcast()`` call refits ~10 PoissonDC models and would dominate request
latency without it.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import date
from pathlib import Path as FilePath
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Path, Request
from pydantic import BaseModel, Field, ValidationError

from wc2026.db.session import get_engine
from wc2026.eval.backtest import HindcastConfig, hindcast
from wc2026.eval.calibration import (
    aggregate,
    base_rates,
    baseline_log_loss,
    reliability_diagram,
)
from wc2026.eval.rolling import RollingCalibration, compute_rolling
from wc2026.ingest.football_data_org import load_wc_match_id_map
from wc2026.ingest.kaggle_intl import load_played

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/track-record")


class PerMatchCalibrationRow(BaseModel):
    match_date: date
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    p_home: float
    p_draw: float
    p_away: float
    observed: str = Field(description="H / D / A")
    log_loss: float
    brier: float
    rps: float
    model_version: str


class WC2026TrackRecord(BaseModel):
    """Aggregate + per-match WC 2026 calibration as of the request time."""

    n_completed: int
    log_loss: float | None
    brier: float | None
    rps: float | None
    per_match: list[PerMatchCalibrationRow]


def _serialize(calibration: RollingCalibration) -> WC2026TrackRecord:
    return WC2026TrackRecord(
        n_completed=calibration.n_completed,
        log_loss=calibration.log_loss,
        brier=calibration.brier,
        rps=calibration.rps,
        per_match=[PerMatchCalibrationRow(**row.__dict__) for row in calibration.per_match],
    )


@router.get("/wc2026", response_model=WC2026TrackRecord)
def wc2026_track_record(request: Request) -> WC2026TrackRecord:
    _ = request  # FastAPI passes the request; we don't currently need it.
    mapping = load_wc_match_id_map()
    try:
        eng = get_engine()
        result = compute_rolling(match_id_to_fixture=mapping, engine=eng)
    except Exception as exc:
        logger.exception("wc2026 track-record DB query failed")
        raise HTTPException(
            status_code=503,
            detail=f"WC2026 track-record DB query failed: {exc.__class__.__name__}",
        ) from exc
    return _serialize(result)


# --- Historical hindcasts -------------------------------------------------

_HISTORICAL_SPANS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "WC2022": (pd.Timestamp("2022-11-20"), pd.Timestamp("2022-12-18")),
    "WC2018": (pd.Timestamp("2018-06-14"), pd.Timestamp("2018-07-15")),
}

_BOOKMAKER_LITERATURE: dict[str, dict[str, Any]] = {
    "WC2018": {
        "log_loss_low": 0.96,
        "log_loss_high": 1.00,
        "cite": "Wheatcroft 2019 (RPS=0.181); Constantinou 2019 (log-loss range)",
    },
    "WC2022": {
        "log_loss_low": 0.95,
        "log_loss_high": 1.00,
        "cite": "estimate from market consensus (no peer-reviewed aggregate yet)",
    },
}

HISTORICAL_CACHE_TTL_SECONDS = 24 * 3600
_HISTORICAL_CACHE: dict[str, tuple[float, HistoricalTrackRecord]] = {}
_HISTORICAL_CACHE_LOCK = threading.Lock()


class ReliabilityBinResponse(BaseModel):
    outcome: str = Field(description="H / D / A")
    bin_low: float
    bin_high: float
    n: int
    mean_predicted: float
    realized_frequency: float


class HistoricalHeadline(BaseModel):
    n_matches: int
    log_loss: float
    brier: float
    rps: float
    baseline_log_loss: float
    base_h: float
    base_d: float
    base_a: float


class BookmakerReference(BaseModel):
    log_loss_low: float
    log_loss_high: float
    cite: str


class HistoricalTrackRecord(BaseModel):
    tournament: str = Field(description="WC2018 or WC2022")
    headline: HistoricalHeadline
    reliability: list[ReliabilityBinResponse]
    bookmaker_reference: BookmakerReference | None = None


def _compute_historical(tournament: str) -> HistoricalTrackRecord:
    """Day-by-day hindcast + reliability bins for ``tournament``.

    Heavy: refits ~10 PoissonDC models. Caller must cache the result; this
    function does NOT touch the module-level cache.
    """
    if tournament not in _HISTORICAL_SPANS:
        raise HTTPException(
            status_code=404,
            detail=f"unknown tournament {tournament!r}; valid: {sorted(_HISTORICAL_SPANS)}",
        )
    start, end = _HISTORICAL_SPANS[tournament]
    history = load_played()
    target = history[
        (history["tournament"] == "FIFA World Cup")
        & (history["date"] >= start)
        & (history["date"] <= end)
    ].copy()
    preds = hindcast(target, history, cfg=HindcastConfig())
    clean = preds.dropna(subset=["p_home", "p_draw", "p_away", "observed"])
    metrics = aggregate(clean)
    obs = clean["observed"].tolist()
    rates = base_rates(obs)
    bins = [
        ReliabilityBinResponse(
            outcome=b.outcome,
            bin_low=b.bin_low,
            bin_high=b.bin_high,
            n=b.n,
            mean_predicted=b.mean_predicted,
            realized_frequency=b.realized_frequency,
        )
        for b in reliability_diagram(clean, n_bins=10)
        if b.n > 0
    ]
    book = _BOOKMAKER_LITERATURE.get(tournament)
    return HistoricalTrackRecord(
        tournament=tournament,
        headline=HistoricalHeadline(
            n_matches=metrics.n,
            log_loss=metrics.log_loss,
            brier=metrics.brier,
            rps=metrics.rps,
            baseline_log_loss=baseline_log_loss(obs),
            base_h=rates["H"],
            base_d=rates["D"],
            base_a=rates["A"],
        ),
        reliability=bins,
        bookmaker_reference=BookmakerReference(**book) if book is not None else None,
    )


@router.get(
    "/historical/{tournament}",
    response_model=HistoricalTrackRecord,
    description=(
        "Headline calibration metrics + per-outcome reliability bins for a "
        "completed World Cup. Cached for 24 hours per (tournament); the "
        "underlying hindcast refits ~10 PoissonDC models so we run it once "
        "per day and serve every dashboard request from the cache."
    ),
)
def historical_track_record(
    tournament: str = Path(
        ...,
        pattern=r"^WC(2018|2022)$",
        description="One of WC2018 / WC2022.",
    ),
) -> HistoricalTrackRecord:
    now = time.monotonic()
    with _HISTORICAL_CACHE_LOCK:
        cached = _HISTORICAL_CACHE.get(tournament)
        if cached is not None and now - cached[0] < HISTORICAL_CACHE_TTL_SECONDS:
            return cached[1]
    try:
        payload = _compute_historical(tournament)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("historical track-record failed for %s", tournament)
        raise HTTPException(
            status_code=503,
            detail=f"historical hindcast failed: {exc.__class__.__name__}",
        ) from exc
    with _HISTORICAL_CACHE_LOCK:
        _HISTORICAL_CACHE[tournament] = (now, payload)
    return payload


# --- Club-football bookmaker benchmark ------------------------------------

BOOKMAKER_BENCHMARK_PATH = FilePath("data/artifacts/bookmaker_benchmark/latest.json")


class BookmakerBenchmark(BaseModel):
    """Club-football PoissonDC vs Bet365/Pinnacle closing odds.

    The corpus is ``football-data.co.uk`` league CSVs (no WC odds in any
    free aggregate), so this is a *structural* check on the model
    architecture, not a WC-specific calibration claim. The
    ``cite`` on the WC reference (literature constants) covers that gap.
    """

    as_of: str = Field(description="ISO-8601 timestamp the artifact was written.")
    cutoff: str = Field(description="Holdout cutoff date (matches on/after are test set).")
    n_train: int
    n_test: int
    n_scored: int
    poisson_log_loss: float | None
    bookmaker_log_loss: float | None
    delta: float | None = Field(
        description="Our PoissonDC log-loss minus the bookmaker log-loss. Negative = we beat the market.",
    )
    leagues: list[tuple[str, str]] = Field(
        description="(season_code, league_code) pairs included in the corpus.",
    )
    half_life_days: float


@router.get(
    "/bookmaker-benchmark",
    response_model=BookmakerBenchmark,
    description=(
        "Reads the artifact written by ``scripts/backtest_against_bookmaker.py`` "
        "and exposes it to the dashboard. Returns 404 when the artifact is "
        "missing (the script hasn't been run yet)."
    ),
)
def bookmaker_benchmark() -> BookmakerBenchmark:
    if not BOOKMAKER_BENCHMARK_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "bookmaker benchmark artifact not on disk; run "
                "`uv run python scripts/backtest_against_bookmaker.py` to generate it."
            ),
        )
    try:
        payload = json.loads(BOOKMAKER_BENCHMARK_PATH.read_text())
    except json.JSONDecodeError as exc:
        logger.exception("bookmaker benchmark artifact is not valid JSON")
        raise HTTPException(
            status_code=503,
            detail=f"bookmaker benchmark artifact malformed: {exc.msg}",
        ) from exc
    try:
        return BookmakerBenchmark(**payload)
    except ValidationError as exc:
        logger.exception(
            "bookmaker benchmark artifact failed pydantic validation"
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "bookmaker benchmark artifact has an unexpected shape; "
                "regenerate via scripts/backtest_against_bookmaker.py"
            ),
        ) from exc


__all__ = ["router"]
