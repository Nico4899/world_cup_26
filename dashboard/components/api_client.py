"""Thin httpx wrapper around the FastAPI app for dashboard pages."""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

API_URL = os.environ.get("WC2026_API_URL", "http://localhost:8000")
TIMEOUT_SECONDS = 15.0


class APIUnreachable(RuntimeError):
    pass


def get_json(path: str, params: dict[str, Any] | None = None) -> Any:
    """GET ``API_URL + path`` and return parsed JSON. Raises APIUnreachable on connection failure."""
    try:
        r = httpx.get(f"{API_URL}{path}", params=params, timeout=TIMEOUT_SECONDS)
    except httpx.HTTPError as exc:
        raise APIUnreachable(f"could not reach API at {API_URL}: {exc}") from exc
    r.raise_for_status()
    return r.json()


def render_unreachable_warning(exc: APIUnreachable) -> None:
    """Common UI fragment when the API isn't responding."""
    st.warning(
        f"⚠️ Couldn't reach the prediction API at `{API_URL}`.\n\n"
        "Start it locally with: `uv run uvicorn wc2026.api.main:app`.\n\n"
        f"Underlying error: `{exc}`"
    )
    st.button("Retry", on_click=st.rerun)


@st.cache_data(ttl=300, show_spinner="Loading fixtures…")
def list_matches(date: str | None = None, group: str | None = None) -> list[dict[str, Any]]:
    params: dict[str, Any] = {}
    if date:
        params["date"] = date
    if group:
        params["group"] = group
    return get_json("/api/v1/matches", params=params or None)


@st.cache_data(ttl=300, show_spinner="Loading match…")
def get_match(match_id: int) -> dict[str, Any]:
    """Returns ``{"fixture": ..., "prediction": ...}`` for a single fixture.

    The prediction includes the full score_matrix (top-5 scorelines + 11×11 joint matrix)
    so callers don't need a separate ``/api/v1/predictions/...`` round-trip.
    """
    return get_json(f"/api/v1/matches/{match_id}")


@st.cache_data(ttl=300, show_spinner="Computing prediction…")
def get_prediction(home: str, away: str, neutral: bool = True) -> dict[str, Any]:
    """Pairwise prediction with full score_matrix; useful for non-fixture matchups."""
    return get_json(f"/api/v1/predictions/{home}/{away}", params={"neutral": str(neutral).lower()})
