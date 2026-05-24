"""Crest / kit / stadium lookups for the dashboard.

Loads from ``/api/v1/teams/{team}/assets`` and caches per-team results for the
session. When the API returns all-null fields (no TheSportsDB row), the
``render_team_chip`` helper falls back to a plain text label so missing assets
don't blow up the layout.
"""

from __future__ import annotations

import httpx
import streamlit as st

from dashboard.components.api_client import APIUnreachable, get_json


@st.cache_data(ttl=3600, show_spinner=False)
def get_team_assets(team: str) -> dict:
    """Return ``{crest_url, kit_home_color, ...}`` for ``team``.

    Cached for 1 hour because the upstream is weekly. Falls back to an
    all-null payload (with the team name) when the API is unreachable or
    returns 503 — UI callers always get a valid dict.
    """
    try:
        return get_json(f"/api/v1/teams/{team}/assets")
    except APIUnreachable:
        return {"team": team}
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            # The pre-Phase-9 API didn't have /assets.
            return {"team": team}
        if exc.response.status_code == 503:
            return {"team": team}
        raise


def crest_img_html(assets: dict, *, height_px: int = 24) -> str:
    """Return a tiny ``<img>`` tag for the crest, or '' if there's no URL."""
    url = (assets or {}).get("crest_url")
    if not url:
        return ""
    return (
        f"<img src='{url}' alt='' style='height:{height_px}px;vertical-align:middle;"
        f"margin-right:6px;border-radius:3px'/>"
    )


def render_team_chip(team: str, *, bold: bool = False) -> str:
    """Inline-renderable HTML chip: crest (if available) + team name.

    Designed for use inside ``st.markdown(..., unsafe_allow_html=True)``.
    Always returns a usable string even when no asset row exists.
    """
    assets = get_team_assets(team)
    crest = crest_img_html(assets)
    name = f"<strong>{team}</strong>" if bold else team
    return f"{crest}{name}"


def render_versus_header(home: str, away: str) -> str:
    """A 'crest · Home vs Away · crest' header for match-detail-style views."""
    home_html = render_team_chip(home, bold=True)
    away_html = render_team_chip(away, bold=True)
    return (
        f"<div style='display:flex;align-items:center;gap:12px;font-size:1.1rem'>"
        f"{home_html} <span style='color:#888'>vs</span> {away_html}"
        f"</div>"
    )


__all__ = [
    "crest_img_html",
    "get_team_assets",
    "render_team_chip",
    "render_versus_header",
]
