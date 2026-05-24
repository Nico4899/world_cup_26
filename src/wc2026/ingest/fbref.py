"""Ingest team-level match logs with xG from FBref (sports-reference.com).

Why
---
StatsBomb open data only covers a handful of competitions (WC 2018, WC 2022,
Euros 2020 + 2024). For *recent form* — friendlies, World Cup qualifiers,
Nations League — we need a wider source. FBref's national-team pages publish
match-by-match xG figures and are scrape-tolerated for personal/research use.

We respect a polite 1 req / 3 s rate limit and cache responses to SQLite via
``requests-cache``. We also strip HTML comments before parsing: FBref wraps
many tables in ``<!-- ... -->`` to discourage simple scrapers, but the table
HTML inside is well-formed and ``pandas.read_html`` parses it once the
comment markers are gone.

The parsed schema (per row, one row per match) mirrors what we persist in
``raw_match_xg``:

    match_date (Date), competition (str), home_team (str), away_team (str),
    team (str — the team whose page we scraped), opponent (str), venue (str),
    gf (Int64), ga (Int64), xg_for (float64), xg_against (float64)
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import requests_cache

from wc2026.ingest._http import RateLimiter

BASE_URL = "https://fbref.com"

DEFAULT_TARGET = Path("data/raw/fbref")
DEFAULT_CACHE = Path("data/raw/fbref/.http_cache")
DEFAULT_CACHE_EXPIRY_SECONDS = 24 * 3600

RATE_LIMIT_REQUESTS = 1
RATE_LIMIT_WINDOW_SECONDS = 3.0  # be polite — FBref bans aggressive scrapers

USER_AGENT = (
    "wc2026-predictor/0.1 "
    "(+https://github.com/Nico4899/world_cup_26; nico.fliegel@gmail.com) "
    "personal-research; calibrated WC 2026 predictions"
)

_HTML_COMMENT_RE = re.compile(r"<!--|-->")

logger = logging.getLogger(__name__)


_RATE_LIMITER = RateLimiter.make(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)


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
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,*/*"})
    return session


def _strip_html_comments(html: str) -> str:
    """Remove ``<!--`` and ``-->`` markers while keeping the wrapped HTML intact.

    FBref encloses some tables in HTML comments so naive scrapers miss them;
    once the markers are gone, ``pd.read_html`` finds the tables normally.
    """
    return _HTML_COMMENT_RE.sub("", html)


def _find_match_log_table(html: str) -> pd.DataFrame | None:
    """Return the FBref match-log table (Date / Comp / xG / xGA columns)."""
    cleaned = _strip_html_comments(html)
    try:
        tables = pd.read_html(StringIO(cleaned))
    except (ValueError, ImportError):
        return None
    for original in tables:
        # FBref tables use MultiIndex columns when they have category groups;
        # flatten before testing.
        tbl = original
        if isinstance(tbl.columns, pd.MultiIndex):
            tbl = tbl.copy()
            tbl.columns = [col[-1] if isinstance(col, tuple) else col for col in tbl.columns]
        cols = {str(c).strip().lower() for c in tbl.columns}
        if {"date", "xg", "xga"}.issubset(cols):
            return tbl
    return None


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case + strip column names; collapse MultiIndex if present."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    return df


def parse_match_log_html(html: str, *, team: str) -> pd.DataFrame:
    """Parse a FBref team match-log page into a per-match DataFrame.

    Filters out the FBref "summary" trailer rows (where Date is non-date), and
    rows lacking either xG or xGA.

    Returns columns:
        match_date (datetime64[ns]), competition (str), opponent (str),
        venue (str), team (str), gf (Int64), ga (Int64), xg_for (float64),
        xg_against (float64)
    """
    table = _find_match_log_table(html)
    if table is None or table.empty:
        return pd.DataFrame(
            columns=[
                "match_date",
                "competition",
                "opponent",
                "venue",
                "team",
                "gf",
                "ga",
                "xg_for",
                "xg_against",
            ]
        )
    df = _normalise_columns(table)
    # Some columns FBref publishes: date, time, comp, round, day, venue,
    # result, gf, ga, opponent, xg, xga, ...
    df = df.copy()
    df["match_date"] = pd.to_datetime(df.get("date"), errors="coerce")
    df = df.dropna(subset=["match_date"])
    df = df[df["xg"].notna() & df["xga"].notna()]
    out = pd.DataFrame(
        {
            "match_date": df["match_date"],
            "competition": df.get("comp", "").astype(str),
            "opponent": df.get("opponent", "").astype(str),
            "venue": df.get("venue", "").astype(str),
            "team": team,
            "gf": pd.to_numeric(df.get("gf"), errors="coerce").astype("Int64"),
            "ga": pd.to_numeric(df.get("ga"), errors="coerce").astype("Int64"),
            "xg_for": pd.to_numeric(df["xg"], errors="coerce"),
            "xg_against": pd.to_numeric(df["xga"], errors="coerce"),
        }
    )
    return out.reset_index(drop=True)


def fetch_match_log(
    url: str,
    *,
    team: str,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """GET an FBref team match-log page and parse it."""
    sess = session or _make_session()
    _RATE_LIMITER.acquire()
    resp = sess.get(url, timeout=30)
    resp.raise_for_status()
    return parse_match_log_html(resp.text, team=team)


def fetch_team_match_logs(
    team_pages: Iterable[tuple[str, str]],
    *,
    session: requests.Session | None = None,
    target_dir: Path = DEFAULT_TARGET,
    today: datetime | None = None,
) -> Path:
    """Fetch match logs for every (team_name, page_url) tuple → combined Parquet.

    Writes ``target_dir / fbref_xg_<YYYY-MM-DD>.parquet`` and returns the path.
    """
    today = today or datetime.now(UTC)
    target_dir.mkdir(parents=True, exist_ok=True)
    sess = session or _make_session()
    frames: list[pd.DataFrame] = []
    for team, url in team_pages:
        try:
            df = fetch_match_log(url, team=team, session=sess)
        except requests.HTTPError as exc:
            logger.warning("FBref match log for %s (%s) failed: %s", team, url, exc)
            continue
        if not df.empty:
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out = target_dir / f"fbref_xg_{today:%Y-%m-%d}.parquet"
    combined.to_parquet(out, index=False)
    return out


def load_latest_snapshot(target_dir: Path = DEFAULT_TARGET) -> pd.DataFrame:
    paths = sorted(target_dir.glob("fbref_xg_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No fbref_xg_*.parquet snapshots in {target_dir}")
    return pd.read_parquet(paths[-1])


__all__ = [
    "BASE_URL",
    "DEFAULT_CACHE",
    "DEFAULT_TARGET",
    "RATE_LIMIT_REQUESTS",
    "RATE_LIMIT_WINDOW_SECONDS",
    "fetch_match_log",
    "fetch_team_match_logs",
    "load_latest_snapshot",
    "parse_match_log_html",
]
