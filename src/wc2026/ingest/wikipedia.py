"""Ingest tournament squads + FIFA Men's Ranking from Wikipedia.

Two distinct data shapes are handled in this module:

1. **Tournament squads** — pages titled e.g. "Argentina at the 2026 FIFA World
   Cup" carry squad lists built from the
   ``{{nat fs g player|no=…|pos=…|name=…|caps=…|goals=…|club=…}}`` template
   family. We fetch raw wikitext via the MediaWiki API and parse the
   templates into rows.

2. **FIFA Men's World Ranking** — the eponymous Wikipedia page has a current-
   rankings table (Rank / Team / Points / change-since-previous). We parse the
   page HTML with ``pandas.read_html`` and produce one row per team.

Both ingesters are conservative: rows that can't be parsed (unusual template
variants, footnoted ranks, etc.) are skipped with a warning rather than aborting
the run. The model does not depend on either output yet — they feed Phase 4 and
Phase 9 features and UI surfaces.
"""

from __future__ import annotations

import io
import logging
import re
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import requests
import requests_cache

API_URL = "https://en.wikipedia.org/w/api.php"
FIFA_RANKING_URL = "https://en.wikipedia.org/wiki/FIFA_Men%27s_World_Ranking"

DEFAULT_TARGET = Path("data/raw/wikipedia")
DEFAULT_CACHE = Path("data/raw/wikipedia/.http_cache")
DEFAULT_CACHE_EXPIRY_SECONDS = 24 * 3600

USER_AGENT = (
    "wc2026-predictor/0.1 "
    "(+https://github.com/Nico4899/world_cup_26; nico.fliegel@gmail.com) "
    "personal-research; calibrated WC 2026 predictions"
)

_PLAYER_TEMPLATE_START_RE = re.compile(
    r"\{\{\s*nat fs (?:g|r) player\b",
    flags=re.IGNORECASE,
)
_BIRTH_DATE_RE = re.compile(
    r"\{\{\s*Birth date(?:\s+and\s+age2?)?\s*\|(?P<args>[^{}]+?)\}\}",
    flags=re.IGNORECASE | re.DOTALL,
)

logger = logging.getLogger(__name__)


def _iter_template_bodies(wikitext: str) -> list[str]:
    """Yield the body of every ``{{nat fs g|r player ...}}`` template.

    Uses a brace-depth walker so nested templates (``{{Birth date|...}}``)
    do not confuse the matcher.
    """
    bodies: list[str] = []
    pos = 0
    while True:
        match = _PLAYER_TEMPLATE_START_RE.search(wikitext, pos)
        if not match:
            break
        depth = 1
        i = match.end()
        n = len(wikitext)
        while i < n - 1 and depth > 0:
            if wikitext[i] == "{" and wikitext[i + 1] == "{":
                depth += 1
                i += 2
                continue
            if wikitext[i] == "}" and wikitext[i + 1] == "}":
                depth -= 1
                if depth == 0:
                    break
                i += 2
                continue
            i += 1
        if depth != 0:
            # Malformed template; stop here to avoid infinite loop.
            break
        bodies.append(wikitext[match.end() : i])
        pos = i + 2
    return bodies


def _split_template_args(body: str) -> list[str]:
    """Split a template body on ``|`` at brace-depth zero.

    Honours nested ``{{...}}`` so e.g. an ``age={{Birth date|1990|5|10}}``
    parameter is not chopped on the inner ``|``.
    """
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    i = 0
    n = len(body)
    while i < n:
        ch = body[i]
        if ch == "{" and i + 1 < n and body[i + 1] == "{":
            depth += 1
            buf.append("{{")
            i += 2
            continue
        if ch == "}" and i + 1 < n and body[i + 1] == "}":
            depth -= 1
            buf.append("}}")
            i += 2
            continue
        if ch == "|" and depth == 0:
            parts.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts


def _parse_template_kwargs(body: str) -> dict[str, str]:
    """Split a Wikipedia template body on ``|`` into a ``name=value`` dict.

    Skips positional args (entries without ``=``).
    """
    out: dict[str, str] = {}
    for raw in _split_template_args(body):
        part = raw.strip()
        if not part or "=" not in part:
            continue
        k, _, v = part.partition("=")
        out[k.strip().lower()] = v.strip()
    return out


def _extract_birth_date(value: str) -> date | None:
    """Extract YYYY-MM-DD from a ``{{Birth date|YYYY|M|D}}`` substring."""
    if not value:
        return None
    match = _BIRTH_DATE_RE.search(value)
    if not match:
        return None
    parts = [p.strip() for p in match.group("args").split("|")]
    # Birth-date template historically had (year, month, day) in the first 3
    # positions; later variants may carry a `df=y` flag — strip non-numeric.
    numeric = [int(p) for p in parts if p.isdigit()]
    if len(numeric) < 3:
        return None
    # The two known shapes are (year, month, day, ...) and (current_year,
    # current_month, current_day, birth_year, birth_month, birth_day) for
    # `birth date and age2`. We pick the last three as birth date in that case.
    if len(numeric) >= 6:
        y, m, d = numeric[3], numeric[4], numeric[5]
    else:
        y, m, d = numeric[0], numeric[1], numeric[2]
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _coerce_int(value: str | None) -> int | None:
    if not value:
        return None
    digits = re.findall(r"-?\d+", value)
    return int(digits[0]) if digits else None


def parse_squad_wikitext(
    wikitext: str,
    *,
    team: str,
    tournament: str,
    snapshot_date: date,
) -> pd.DataFrame:
    """Extract player rows from a Wikipedia tournament-squad page.

    Returns a DataFrame with columns matching ``RawSquad``:
        tournament, team, player_name, snapshot_date, shirt_number, position,
        birth_date, club, caps, goals
    """
    rows: list[dict[str, object]] = []
    for body in _iter_template_bodies(wikitext):
        kwargs = _parse_template_kwargs(body)
        name = kwargs.get("name")
        if not name:
            continue
        rows.append(
            {
                "tournament": tournament,
                "team": team,
                "player_name": name,
                "snapshot_date": snapshot_date,
                "shirt_number": _coerce_int(kwargs.get("no")),
                "position": (kwargs.get("pos") or None),
                "birth_date": _extract_birth_date(kwargs.get("age", "")),
                "club": kwargs.get("club") or None,
                "caps": _coerce_int(kwargs.get("caps")),
                "goals": _coerce_int(kwargs.get("goals")),
            }
        )
    return pd.DataFrame(rows)


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
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json,text/html,*/*"})
    return session


def fetch_squad_wikitext(
    page_title: str,
    *,
    session: requests.Session | None = None,
) -> str:
    """GET the raw wikitext for one squad page via the MediaWiki API."""
    sess = session or _make_session()
    params = {
        "action": "parse",
        "page": page_title,
        "prop": "wikitext",
        "format": "json",
        "formatversion": "2",
    }
    resp = sess.get(API_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    parse_block = payload.get("parse") or {}
    wikitext = parse_block.get("wikitext")
    if isinstance(wikitext, dict):
        # legacy MediaWiki shape {"wikitext": {"*": "..."}}
        wikitext = wikitext.get("*", "")
    return wikitext or ""


def fetch_all_squads(
    team_pages: dict[str, str],
    *,
    tournament: str = "FIFA World Cup 2026",
    target_dir: Path = DEFAULT_TARGET,
    session: requests.Session | None = None,
    today: datetime | None = None,
) -> Path:
    """Fetch and parse squads for ``team_pages`` (canonical team name → page title).

    Writes ``squads_<tournament_slug>_<YYYY-MM-DD>.parquet`` and returns its path.
    """
    today = today or datetime.now(UTC)
    snapshot_date = today.date()
    target_dir.mkdir(parents=True, exist_ok=True)
    sess = session or _make_session()
    frames: list[pd.DataFrame] = []
    for team, page_title in team_pages.items():
        try:
            wikitext = fetch_squad_wikitext(page_title, session=sess)
        except requests.HTTPError as exc:
            logger.warning("Wikipedia squad fetch for %r failed: %s", page_title, exc)
            continue
        df = parse_squad_wikitext(
            wikitext, team=team, tournament=tournament, snapshot_date=snapshot_date
        )
        if df.empty:
            logger.warning("No player templates parsed from %r", page_title)
            continue
        frames.append(df)
    combined = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["team", "player_name"])
    )
    slug = tournament.lower().replace(" ", "_")
    out = target_dir / f"squads_{slug}_{today:%Y-%m-%d}.parquet"
    combined.to_parquet(out, index=False)
    return out


def _find_rankings_table(html: str) -> pd.DataFrame | None:
    """Return the first table in the page HTML that looks like the FIFA ranking."""
    try:
        tables = pd.read_html(io.StringIO(html))
    except (ValueError, ImportError):
        # ValueError: pandas found no parseable tables.
        # ImportError: pandas tried html5lib first; if absent and lxml can't
        # parse the HTML either, treat as "no tables."
        return None
    for tbl in tables:
        cols = {str(c).strip().lower() for c in tbl.columns}
        # Accept any table that has rank+team+points (the actual current-rankings
        # table has additional columns we can ignore).
        if {"rank", "team", "points"}.issubset(cols):
            return tbl
    return None


def parse_fifa_ranking_html(html: str, *, ranking_date: date) -> pd.DataFrame:
    """Extract the current-rankings table from a FIFA-ranking page snapshot.

    Returns a DataFrame matching ``RawFifaRanking``:
        ranking_date, team, rank, points, previous_rank
    """
    table = _find_rankings_table(html)
    if table is None or table.empty:
        return pd.DataFrame(columns=["ranking_date", "team", "rank", "points", "previous_rank"])
    col_map = {str(c).strip().lower(): c for c in table.columns}
    df = pd.DataFrame(
        {
            "ranking_date": ranking_date,
            "team": table[col_map["team"]].astype(str).str.strip(),
            "rank": pd.to_numeric(table[col_map["rank"]], errors="coerce").astype("Int64"),
            "points": pd.to_numeric(table[col_map["points"]], errors="coerce"),
        }
    )
    # "Previous" / "Last" / "Prev" — be lenient.
    prev_key = next(
        (col_map[k] for k in ("previous", "previous rank", "last", "prev") if k in col_map),
        None,
    )
    if prev_key is not None:
        df["previous_rank"] = pd.to_numeric(table[prev_key], errors="coerce").astype("Int64")
    else:
        df["previous_rank"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    df = df.dropna(subset=["rank"]).reset_index(drop=True)
    return df


def fetch_fifa_ranking(
    *,
    session: requests.Session | None = None,
    today: datetime | None = None,
    target_dir: Path = DEFAULT_TARGET,
) -> Path:
    """Fetch the FIFA Men's World Ranking Wikipedia page, parse, write Parquet.

    The ``ranking_date`` in the resulting Parquet is *today* — we don't try to
    parse the precise "as of" date from the page header (the table is updated
    in lock-step with the official ranking, so today ± 1 day is fine for our
    monthly cadence).
    """
    today = today or datetime.now(UTC)
    target_dir.mkdir(parents=True, exist_ok=True)
    sess = session or _make_session()
    resp = sess.get(FIFA_RANKING_URL, timeout=30)
    resp.raise_for_status()
    df = parse_fifa_ranking_html(resp.text, ranking_date=today.date())
    out = target_dir / f"fifa_ranking_{today:%Y-%m-%d}.parquet"
    df.to_parquet(out, index=False)
    return out


__all__ = [
    "API_URL",
    "DEFAULT_CACHE",
    "DEFAULT_TARGET",
    "FIFA_RANKING_URL",
    "fetch_all_squads",
    "fetch_fifa_ranking",
    "fetch_squad_wikitext",
    "parse_fifa_ranking_html",
    "parse_squad_wikitext",
]
