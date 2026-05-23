"""Ingest national-team crest / kit / stadium metadata from TheSportsDB.

Docs: https://www.thesportsdb.com/api.php

API key
-------
Set ``THESPORTSDB_API_KEY`` in the environment. The site documents ``123`` as
the public test key (used as the default here). Free-tier limit is 30
req/minute; the Patreon $0 tier raises it to 100/minute. Both are rate-limited
client-side via the sliding window in this module.

Why this source
---------------
We use TheSportsDB strictly for **UI assets** (crest URL, kit colours, stadium
metadata). The model does not consume any field this ingester produces; if the
ingester is broken or skipped the predictions still run, the dashboard just
falls back to plain text flags.

The site is community-edited, so a couple of name spellings differ from our
canonical (Jürisoo-derived) team names; the ``DEFAULT_ALIASES`` map below
covers the cases known at write time.

Parsed schema
-------------
``fetch_team_assets`` writes a Parquet snapshot to
``data/raw/thesportsdb/teams_<YYYY-MM-DD>.parquet`` with the columns:

    team (str, canonical), thesportsdb_id (Int64), crest_url (str | None),
    kit_home_color (str | None), kit_away_color (str | None),
    stadium_name (str | None), stadium_capacity (Int64),
    stadium_city (str | None), stadium_country (str | None),
    fetched_at (datetime64[ns, UTC])
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import requests_cache

BASE_URL_TEMPLATE = "https://www.thesportsdb.com/api/v1/json/{api_key}"
DEFAULT_API_KEY = "123"  # public test key per TheSportsDB docs

DEFAULT_TARGET = Path("data/raw/thesportsdb")
DEFAULT_CACHE = Path("data/raw/thesportsdb/.http_cache")
DEFAULT_CACHE_EXPIRY_SECONDS = 7 * 24 * 3600  # weekly refresh cadence

ENV_API_KEY = "THESPORTSDB_API_KEY"
RATE_LIMIT_REQUESTS = 30
RATE_LIMIT_WINDOW_SECONDS = 60.0

USER_AGENT = (
    "wc2026-predictor/0.1 "
    "(+https://github.com/Nico4899/world_cup_26; nico.fliegel@gmail.com) "
    "personal-research; calibrated WC 2026 predictions"
)

# Canonical (Jürisoo-derived) name → TheSportsDB lookup string.
# Cases where the upstream's spelling differs from ours go here.
DEFAULT_ALIASES: dict[str, str] = {
    "United States": "USA",
    "South Korea": "Korea Republic",
    "Ivory Coast": "Côte d'Ivoire",
    "Cape Verde": "Cape Verde Islands",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Czech Republic": "Czechia",
}

logger = logging.getLogger(__name__)


class MissingApiKeyError(RuntimeError):
    """Raised when no API key is configured (only relevant on a paid plan)."""


@dataclass(frozen=True)
class _RateLimiter:
    """Sliding-window limiter — copy of the one in football_data_org for isolation."""

    limit: int
    window: float
    _calls: deque[float]
    _lock: threading.Lock

    @classmethod
    def make(cls, limit: int, window: float) -> _RateLimiter:
        return cls(limit=limit, window=window, _calls=deque(), _lock=threading.Lock())

    def acquire(self, *, now: float | None = None, sleep=time.sleep) -> None:
        with self._lock:
            t = now if now is not None else time.monotonic()
            while self._calls and t - self._calls[0] >= self.window:
                self._calls.popleft()
            if len(self._calls) >= self.limit:
                wait = self.window - (t - self._calls[0]) + 0.01
                sleep(max(wait, 0.0))
                t2 = time.monotonic()
                while self._calls and t2 - self._calls[0] >= self.window:
                    self._calls.popleft()
                self._calls.append(t2)
            else:
                self._calls.append(t)


_RATE_LIMITER = _RateLimiter.make(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)


def _api_key(api_key: str | None = None) -> str:
    if api_key is not None:
        return api_key
    return os.environ.get(ENV_API_KEY, DEFAULT_API_KEY)


def _make_session(
    cache_path: Path | None = DEFAULT_CACHE,
    expire_seconds: int = DEFAULT_CACHE_EXPIRY_SECONDS,
) -> requests.Session:
    if cache_path is None:
        session: requests.Session = requests.Session()
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        session = requests_cache.CachedSession(
            cache_name=str(cache_path),
            backend="sqlite",
            expire_after=expire_seconds,
            allowable_codes=(200,),
            stale_if_error=True,
        )
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return session


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "" or value == "0":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def parse_team_lookup_response(
    payload: dict[str, Any] | None,
    *,
    canonical_name: str,
    upstream_name: str | None = None,
) -> dict[str, Any] | None:
    """Extract the canonical asset row from a ``searchteams.php`` response.

    TheSportsDB returns ``{"teams": [...]}`` when matches exist, ``{"teams":
    null}`` otherwise. We pick the first entry whose ``strSport == "Soccer"``
    and whose ``strTeam`` matches the requested upstream name case-insensitively
    when possible; otherwise the first soccer entry. Returns ``None`` if the
    payload has no usable soccer team.

    ``canonical_name`` is the name we store; ``upstream_name`` is what we
    actually queried for (may differ via the alias map).
    """
    if not payload:
        return None
    teams = payload.get("teams")
    if not teams:
        return None
    target = (upstream_name or canonical_name).strip().lower()
    soccer = [t for t in teams if (t.get("strSport") or "").lower() == "soccer"]
    if not soccer:
        return None
    exact = [t for t in soccer if (t.get("strTeam") or "").strip().lower() == target]
    team = exact[0] if exact else soccer[0]
    return {
        "team": canonical_name,
        "thesportsdb_id": _coerce_int(team.get("idTeam")),
        "crest_url": _coerce_str(team.get("strBadge") or team.get("strTeamBadge")),
        "kit_home_color": _coerce_str(team.get("strKitColour1")),
        "kit_away_color": _coerce_str(team.get("strKitColour2")),
        "stadium_name": _coerce_str(team.get("strStadium")),
        "stadium_capacity": _coerce_int(team.get("intStadiumCapacity")),
        "stadium_city": _coerce_str(team.get("strStadiumLocation")),
        "stadium_country": _coerce_str(team.get("strCountry")),
    }


def fetch_team(
    team_name: str,
    *,
    api_key: str | None = None,
    session: requests.Session | None = None,
    aliases: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Fetch one team's metadata. Returns None if TheSportsDB has no match."""
    key = _api_key(api_key)
    sess = session or _make_session()
    alias_map = aliases if aliases is not None else DEFAULT_ALIASES
    upstream_name = alias_map.get(team_name, team_name)
    base = BASE_URL_TEMPLATE.format(api_key=key)
    _RATE_LIMITER.acquire()
    resp = sess.get(f"{base}/searchteams.php", params={"t": upstream_name}, timeout=30)
    resp.raise_for_status()
    return parse_team_lookup_response(
        resp.json(), canonical_name=team_name, upstream_name=upstream_name
    )


def fetch_team_assets(
    team_names: Iterable[str],
    *,
    api_key: str | None = None,
    target_dir: Path = DEFAULT_TARGET,
    session: requests.Session | None = None,
    aliases: dict[str, str] | None = None,
    today: datetime | None = None,
) -> Path:
    """Fetch metadata for every team in ``team_names``, write a Parquet snapshot.

    Returns the path to the new file. Teams TheSportsDB has no record of are
    silently skipped (logged at WARNING); a Parquet with zero rows is still
    written if all lookups fail so the scheduler row records the run.
    """
    today = today or datetime.now(UTC)
    target_dir.mkdir(parents=True, exist_ok=True)
    sess = session or _make_session()
    rows: list[dict[str, Any]] = []
    for name in team_names:
        try:
            row = fetch_team(name, api_key=api_key, session=sess, aliases=aliases)
        except requests.HTTPError as exc:
            logger.warning("TheSportsDB lookup for %r failed: %s", name, exc)
            continue
        if row is None:
            logger.warning("TheSportsDB has no soccer entry for %r", name)
            continue
        row["fetched_at"] = today
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.astype(
            {
                "team": "string",
                "thesportsdb_id": "Int64",
                "crest_url": "string",
                "kit_home_color": "string",
                "kit_away_color": "string",
                "stadium_name": "string",
                "stadium_capacity": "Int64",
                "stadium_city": "string",
                "stadium_country": "string",
            }
        )
    out = target_dir / f"teams_{today:%Y-%m-%d}.parquet"
    df.to_parquet(out, index=False)
    return out


def load_latest_snapshot(target_dir: Path = DEFAULT_TARGET) -> pd.DataFrame:
    paths = sorted(target_dir.glob("teams_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No teams_*.parquet snapshots in {target_dir}")
    return pd.read_parquet(paths[-1])


__all__ = [
    "BASE_URL_TEMPLATE",
    "DEFAULT_ALIASES",
    "DEFAULT_API_KEY",
    "ENV_API_KEY",
    "RATE_LIMIT_REQUESTS",
    "RATE_LIMIT_WINDOW_SECONDS",
    "MissingApiKeyError",
    "fetch_team",
    "fetch_team_assets",
    "load_latest_snapshot",
    "parse_team_lookup_response",
]
