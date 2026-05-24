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


def _build_match_id_map(request: Request) -> dict[int, tuple[date, str, str]]:
    """Resolve football-data.org match_id → (date, home, away) via the cached fixtures.

    Returns an empty dict when no cache is on disk; callers then surface an
    empty calibration ("no events ingested yet") rather than 500.
    """
    try:
        from wc2026.ingest.football_data_org import (  # noqa: PLC0415
            WC_COMPETITION_CODE,
            fetch_competition_matches,
        )

        df = fetch_competition_matches(WC_COMPETITION_CODE)
    except Exception:  # broad: caches/keys/network all roll up the same here
        logger.debug("football-data.org cache unavailable for track-record map", exc_info=True)
        return {}
    out: dict[int, tuple[date, str, str]] = {}
    if df.empty:
        return out
    for _, row in df.iterrows():
        match_id = row.get("match_id")
        utc_date = row.get("utc_date")
        home = row.get("home_team")
        away = row.get("away_team")
        if match_id is None or utc_date is None or home is None or away is None:
            continue
        try:
            d = (
                utc_date.date()
                if hasattr(utc_date, "date")
                else date.fromisoformat(str(utc_date)[:10])
            )
            out[int(match_id)] = (d, str(home), str(away))
        except (TypeError, ValueError):
            continue
    # The route also has access to ``request.app.state.feature_sources`` etc.,
    # but football-data.org is the only place the int match_id is defined, so
    # there's nothing else to merge in here.
    _ = request
    return out


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
    mapping = _build_match_id_map(request)
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
