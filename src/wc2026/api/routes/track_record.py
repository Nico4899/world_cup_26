"""WC 2026 rolling calibration endpoint.

``GET /api/v1/track-record/wc2026`` returns ``RollingCalibration`` over every
completed WC 2026 match: aggregate log-loss / Brier / RPS, plus a per-match
diagnostic table. The dashboard Track Record page surfaces this above the
historical WC 2018 + WC 2022 hindcasts.

Implementation note
-------------------
``RawLiveEvent.match_id`` holds the football-data.org numeric id; our
predictions live under the (match_date, home_team, away_team) natural key.
We bridge them by reading the latest cached football-data.org WC fixtures
off disk — no live API call, no Postgres dependency beyond the rows the
live poller already persisted.
"""

from __future__ import annotations

import logging
from datetime import date

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from wc2026.db.session import get_engine
from wc2026.eval.rolling import RollingCalibration, compute_rolling
from wc2026.ingest.football_data_org import load_wc_match_id_map

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


__all__ = ["router"]
