"""'As of ...' freshness banner shown atop every forecast page.

The dashboard spec calls out timestamping as one of the headline interaction
principles: every probability shown must carry a "what state of the world
informed this" annotation. This component centralises that string so the
phrasing stays consistent across Today / Groups / Bracket / Team Profile.

Composition:
- Model fit time + version come from ``/health`` (always available).
- The "after X-Y" tail comes from the most-recent completed WC 2026 match
  pulled from ``/api/v1/track-record/wc2026``. Falls back silently to just
  the model timestamp when no completed matches exist yet (i.e. before
  June 11 2026).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx
import streamlit as st

from dashboard.components.api_client import APIUnreachable, get_json


def _safe_get(path: str) -> dict[str, Any] | None:
    try:
        return get_json(path)
    except (APIUnreachable, httpx.HTTPStatusError):
        return None


def _format_utc(ts: str | None) -> str | None:
    if not ts:
        return None
    try:
        # The API returns ISO strings (Pydantic default). Some have trailing 'Z'.
        cleaned = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
    except (TypeError, ValueError):
        return None
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def _last_completed_summary(track_record: dict[str, Any] | None) -> str | None:
    if not track_record:
        return None
    per_match = track_record.get("per_match") or []
    if not per_match:
        return None
    # The endpoint isn't ordered explicitly; pick the latest match_date row.
    latest = max(
        per_match,
        key=lambda r: (r.get("match_date") or "", r.get("home_team") or ""),
    )
    home = latest.get("home_team")
    away = latest.get("away_team")
    hs = latest.get("home_score")
    as_ = latest.get("away_score")
    if None in (home, away, hs, as_):
        return None
    return f"{home} {hs}-{as_} {away}"


def render_forecast_header() -> None:
    """Render a single-line "As of …" caption.

    The component is best-effort: if the API is unreachable, it stays silent
    rather than rendering an alarming warning (callers usually render their
    own ``render_unreachable_warning`` shortly after).
    """
    health = _safe_get("/health")
    track_record = _safe_get("/api/v1/track-record/wc2026")
    if not health:
        return

    fit_at = _format_utc(health.get("model_fit_at"))
    if not fit_at:
        # No timestamp on disk yet — show nothing rather than a half-truth.
        return

    suffix = ""
    after = _last_completed_summary(track_record)
    if after:
        suffix = f", after {after}"

    model_version = health.get("model_version") or "model"
    st.caption(f"_As of **{fit_at}**{suffix} · {model_version}._")


__all__ = ["render_forecast_header"]
