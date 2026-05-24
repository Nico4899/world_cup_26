"""Ingest competition fixtures and results from football-data.org v4.

Docs: https://www.football-data.org/documentation/api

API key
-------
Set ``FOOTBALL_DATA_ORG_KEY`` in the environment. The free tier permits 10
requests per minute; this module rate-limits to that ceiling and caches GET
responses to ``data/raw/football_data_org/.http_cache.sqlite`` via
``requests-cache`` so re-runs within the cache window are cheap.

If the key is missing, ``fetch_competition_matches`` raises ``MissingApiKeyError``;
tests instead exercise the parser via the bundled fixture file.

Parsed schema
-------------
``fetch_competition_matches`` returns a pandas DataFrame with the columns:
    match_id (Int64), utc_date (datetime64[ns, UTC]), status (str),
    matchday (Int64, nullable), stage (str), group (str, nullable),
    home_team (str), away_team (str), home_score (Int64, nullable),
    away_score (Int64, nullable), winner (str, nullable), venue (str, nullable)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import requests_cache
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from wc2026.ingest._http import RateLimiter

BASE_URL = "https://api.football-data.org/v4"
WC_COMPETITION_CODE = "WC"

DEFAULT_CACHE_DIR = Path("data/raw/football_data_org")
CACHE_FILE = DEFAULT_CACHE_DIR / ".http_cache"
DEFAULT_CACHE_EXPIRY_SECONDS = 6 * 3600

ENV_API_KEY = "FOOTBALL_DATA_ORG_KEY"
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 60.0

USER_AGENT = (
    "wc2026-predictor/0.1 "
    "(+https://github.com/Nico4899/world_cup_26; nico.fliegel@gmail.com) "
    "personal-research; calibrated WC 2026 predictions"
)


class MissingApiKeyError(RuntimeError):
    """Raised when the football-data.org API key is required but unset."""


class RateLimitError(RuntimeError):
    """Raised when the upstream returns HTTP 429; tenacity retries this."""


# Re-exported under the old private name so existing tests
# (``fdo._RateLimiter.make(...)``) keep working without churn.
_RateLimiter = RateLimiter

_RATE_LIMITER = RateLimiter.make(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)


def _build_headers(api_key: str | None) -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if api_key:
        headers["X-Auth-Token"] = api_key
    return headers


def _require_key() -> str:
    key = os.environ.get(ENV_API_KEY)
    if not key:
        raise MissingApiKeyError(
            f"{ENV_API_KEY} is not set. Set it to a football-data.org API token "
            "or use the fixture-driven parser path in tests."
        )
    return key


def _make_cached_session(cache_path: Path = CACHE_FILE) -> requests_cache.CachedSession:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    return requests_cache.CachedSession(
        cache_name=str(cache_path),
        backend="sqlite",
        expire_after=DEFAULT_CACHE_EXPIRY_SECONDS,
        allowable_codes=(200,),
        stale_if_error=True,
    )


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((RateLimitError, httpx.TransportError)),
)
def _http_get_json(
    url: str,
    *,
    api_key: str,
    params: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Polite GET that respects the rate limiter and surfaces 429 as retryable."""
    _RATE_LIMITER.acquire()
    with httpx.Client(timeout=timeout, headers=_build_headers(api_key)) as client:
        resp = client.get(url, params=params)
    if resp.status_code == 429:
        raise RateLimitError(f"429 Too Many Requests from {url}")
    resp.raise_for_status()
    return resp.json()


def parse_competition_matches(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert a /competitions/<code>/matches response into a typed DataFrame.

    Resilient to optional fields: matchday, group, venue, and any of the score
    sub-fields may be absent on scheduled (not-yet-played) fixtures.
    """
    matches = payload.get("matches") or []
    rows: list[dict[str, Any]] = []
    for m in matches:
        score = m.get("score") or {}
        full_time = score.get("fullTime") or {}
        rows.append(
            {
                "match_id": m.get("id"),
                "utc_date": m.get("utcDate"),
                "status": m.get("status"),
                "matchday": m.get("matchday"),
                "stage": m.get("stage"),
                "group": m.get("group"),
                "home_team": (m.get("homeTeam") or {}).get("name"),
                "away_team": (m.get("awayTeam") or {}).get("name"),
                "home_score": full_time.get("home"),
                "away_score": full_time.get("away"),
                "winner": score.get("winner"),
                "venue": m.get("venue"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.astype(
        {
            "match_id": "Int64",
            "status": "string",
            "matchday": "Int64",
            "stage": "string",
            "group": "string",
            "home_team": "string",
            "away_team": "string",
            "home_score": "Int64",
            "away_score": "Int64",
            "winner": "string",
            "venue": "string",
        }
    )
    df["utc_date"] = pd.to_datetime(df["utc_date"], utc=True, errors="coerce")
    return df


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(RateLimitError),
)
def _cached_get_json(
    session: requests_cache.CachedSession,
    url: str,
    *,
    api_key: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Rate-limited cached GET that surfaces 429 as a retryable RateLimitError."""
    _RATE_LIMITER.acquire()
    resp = session.get(url, params=params, headers=_build_headers(api_key), timeout=30)
    if resp.status_code == 429:
        raise RateLimitError(f"429 Too Many Requests from {url}")
    resp.raise_for_status()
    return resp.json()


def fetch_competition_matches(
    competition_id: str = WC_COMPETITION_CODE,
    season: int | None = None,
    *,
    api_key: str | None = None,
    session: requests_cache.CachedSession | None = None,
) -> pd.DataFrame:
    """GET /v4/competitions/{competition_id}/matches?season=YYYY, returns typed DataFrame.

    Raises MissingApiKeyError if no key is configured.
    """
    key = api_key or _require_key()
    sess = session or _make_cached_session()
    url = f"{BASE_URL}/competitions/{competition_id}/matches"
    params: dict[str, Any] = {}
    if season is not None:
        params["season"] = season
    payload = _cached_get_json(sess, url, api_key=key, params=params)
    return parse_competition_matches(payload)


def fetch_match(match_id: int, *, api_key: str | None = None) -> dict[str, Any]:
    """GET /v4/matches/{match_id}, returns the raw JSON dict (uncached)."""
    key = api_key or _require_key()
    return _http_get_json(f"{BASE_URL}/matches/{match_id}", api_key=key)


def load_fixture(path: Path) -> dict[str, Any]:
    """Convenience: load a saved /matches JSON payload from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_wc_match_id_map() -> dict[int, tuple[Any, str, str]]:
    """Best-effort: pull the cached WC fixtures and map ``match_id -> (date, home, away)``.

    Returns ``{}`` if the API key is unset, the cache is empty, or the upstream
    can't be reached — callers degrade gracefully to "no FDO id resolution".
    """
    from datetime import date as _date  # noqa: PLC0415

    try:
        df = fetch_competition_matches(WC_COMPETITION_CODE)
    except Exception:
        # Caches/keys/network errors all roll up the same: no map.
        return {}
    out: dict[int, tuple[Any, str, str]] = {}
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
                else _date.fromisoformat(str(utc_date)[:10])
            )
            out[int(match_id)] = (d, str(home), str(away))
        except (TypeError, ValueError):
            continue
    return out


__all__ = [
    "BASE_URL",
    "ENV_API_KEY",
    "RATE_LIMIT_REQUESTS",
    "RATE_LIMIT_WINDOW_SECONDS",
    "WC_COMPETITION_CODE",
    "MissingApiKeyError",
    "RateLimitError",
    "fetch_competition_matches",
    "fetch_match",
    "load_fixture",
    "load_wc_match_id_map",
    "parse_competition_matches",
]
