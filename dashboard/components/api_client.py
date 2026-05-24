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


def post_json(
    path: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json: Any = None,
) -> Any:
    """POST ``API_URL + path`` and return parsed JSON. Raises APIUnreachable on connection failure."""
    try:
        r = httpx.post(
            f"{API_URL}{path}",
            headers=headers,
            params=params,
            json=json,
            timeout=TIMEOUT_SECONDS,
        )
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


@st.cache_data(ttl=300, show_spinner="Loading recent form…")
def get_recent_form(team: str, n: int = 5) -> list[dict[str, Any]]:
    """Last `n` matches from the team's perspective (W/D/L)."""
    return get_json(f"/api/v1/teams/{team}/recent", params={"n": n})


@st.cache_data(ttl=300, show_spinner="Loading head-to-head…")
def get_h2h(team_a: str, team_b: str, n: int = 10) -> list[dict[str, Any]]:
    """Past matches between `team_a` and `team_b`, date-desc."""
    return get_json(f"/api/v1/h2h/{team_a}/{team_b}", params={"n": n})


@st.cache_data(ttl=600, show_spinner="Running tournament simulation…")
def get_standings(n_sims: int = 2000, seed: int = 42) -> dict[str, Any]:
    """Aggregated Monte Carlo standings (cached 10 min)."""
    return get_json("/api/v1/tournament/standings", params={"n_sims": n_sims, "seed": seed})


def get_live_snapshot(match_id: int) -> dict[str, Any]:
    """Current live state + win-prob for one fixture. Intentionally uncached —
    callers (the Match Detail page during a live match) want fresh state on
    every render."""
    return get_json(f"/api/v1/live/{match_id}")


def get_live_history(match_id: int) -> dict[str, Any]:
    """Full per-event timeline + the latest snapshot. Used by the dashboard's
    live win-prob line chart. Uncached for the same reason as
    :func:`get_live_snapshot`."""
    return get_json(f"/api/v1/live/{match_id}/history")


@st.cache_data(ttl=300, show_spinner="Computing SHAP explanation…")
def get_explanation(
    match_id: int, *, class_name: str = "home_win", top_n: int = 5
) -> dict[str, Any]:
    """SHAP top-features explanation for one WC 2026 fixture (Phase 5 endpoint).

    Returns ``MatchExplanation``-shaped JSON. The caller decides how to
    handle the 503 case (no XGB artefact loaded) — this helper just raises.
    """
    return get_json(
        f"/api/v1/explain/{match_id}",
        params={"class_name": class_name, "top_n": top_n},
    )


@st.cache_data(ttl=3600, show_spinner=False)
def get_team_elo_history(team: str) -> dict[str, Any]:
    """Cached Elo daily snapshots for one team."""
    return get_json(f"/api/v1/teams/{team}/elo-history")


@st.cache_data(ttl=600, show_spinner=False)
def get_team_tournament_probs(team: str) -> dict[str, Any]:
    """Cached per-team advancement probabilities from the latest persisted MC run."""
    return get_json(f"/api/v1/teams/{team}/tournament-probs")
