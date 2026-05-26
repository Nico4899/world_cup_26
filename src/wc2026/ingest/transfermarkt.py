"""Polite Transfermarkt squad-market-value scraper.

Pulls the per-team squad market value as a single Euro number from
``https://www.transfermarkt.com/<team>/startseite/verein/<id>``. The
endpoint is **manual-only** — operators trigger it from the Operator
page; there is no scheduled cron job, by design.

Legal + ethical guardrails
--------------------------
- **User-Agent** identifies the project + a contact email so the site
  maintainer can reach out if anything is wrong.
- **robots.txt is consulted** before every request via
  :class:`urllib.robotparser.RobotFileParser`. The scraper refuses to
  fetch any URL it disallows, even if the call site insists.
- **Caching** is enabled by default — every request is cached to a
  SQLite store for 7 days, so re-running the job doesn't generate new
  upstream load.
- **Personal-research only** — Transfermarkt's T&Cs prohibit commercial
  redistribution. The parquet snapshots this module writes stay on the
  operator's disk; the dashboard surfaces aggregated derived features
  (market-value diff, log market-value diff) but does not expose the
  raw values to end users.

Validation
----------
Müller, Simons & Weinmann (2017), "Beyond Crowd Judgements: Data-driven
estimation of market value in association football" (European Journal
of Operational Research 263.2: 611–624) showed Transfermarkt's
crowd-sourced market values predict transfer fees within ±15%, making
them a reliable team-strength proxy when other signals (Elo, FIFA
ranking) lag squad changes.

Paul Johnson, "Testing Transfermarkt's Squad Market Values" (May 2025)
documents a performance bias: values incorporate recent results, so
they're not a fully independent signal. Treat them as **partially
backward-looking**; the held-out backtest gate (≥0.005 Brier
improvement) protects against double-counting when combined with the
recent-form features.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
import requests_cache

logger = logging.getLogger(__name__)

BASE_URL = "https://www.transfermarkt.com"

USER_AGENT = (
    "wc2026-predictor/0.1 "
    "(+https://github.com/Nico4899/world_cup_26; nico.fliegel@gmail.com) "
    "personal-research; calibrated WC 2026 predictions"
)

DEFAULT_TARGET = Path("data/raw/transfermarkt")
DEFAULT_CACHE = Path("data/raw/transfermarkt/.http_cache")

# ``€nnn.nnnm`` / ``€n.nnbn`` / ``€nnn.nnnk`` — Transfermarkt formats values
# with thousands-separator dots and the suffix "m" (millions), "bn"
# (billions), or "k" (thousands).
_VALUE_RE = re.compile(
    r"€\s*([0-9]+(?:[.,][0-9]+)?)\s*(bn|m|k)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TeamMarketValue:
    """One ``(team_slug, team_name, squad_market_value_eur)`` row."""

    team_slug: str
    team_name: str
    squad_market_value_eur: float
    snapshot_date: pd.Timestamp


def parse_market_value(text: str) -> float | None:
    """Parse a Transfermarkt-formatted market value into a Euro float.

    Examples
    --------
    >>> parse_market_value("€1.05bn")
    1050000000.0
    >>> parse_market_value("€892.50m")
    892500000.0
    >>> parse_market_value("€500k")
    500000.0
    >>> parse_market_value("not a price") is None
    True
    """
    m = _VALUE_RE.search(text)
    if not m:
        return None
    raw, suffix = m.group(1), (m.group(2) or "").lower()
    # Transfermarkt's locale uses a dot as the decimal separator; a comma
    # would be a thousands separator. We've never seen both in one value,
    # but if it ever happens (e.g. "€1,234.56m") the comma-strip is safe.
    normalised = raw.replace(",", "")
    try:
        value = float(normalised)
    except ValueError:
        return None
    multiplier = {"k": 1_000.0, "m": 1_000_000.0, "bn": 1_000_000_000.0, "": 1.0}.get(
        suffix, 1.0
    )
    return value * multiplier


def parse_squad_page(html: str, *, team_slug: str) -> TeamMarketValue | None:
    """Extract the team name + total squad market value from a Transfermarkt page.

    The page's "Total market value" widget is rendered as
    ``Total market value: <a>€892.50m</a>``. We grab the first ``€``-prefixed
    value that survives :func:`parse_market_value` after that label.

    Returns ``None`` when the value can't be parsed (page layout drift / 404
    served as 200 / etc.). Caller decides how loud to fail.
    """
    name_match = re.search(r"<title>([^|<]+?)\s*\|", html)
    team_name = name_match.group(1).strip() if name_match else team_slug
    # Search for the first "Total market value" label then the next "€nnn"
    # token. Fall back to the first "€nnn" anywhere on the page (Transfermarkt
    # English / German label phrasing has drifted in the past).
    label_match = re.search(
        r"Total\s+market\s+value[^€]{0,200}(€[^<]+)",
        html,
        flags=re.IGNORECASE,
    )
    if label_match:
        value = parse_market_value(label_match.group(1))
        if value is not None:
            return TeamMarketValue(
                team_slug=team_slug,
                team_name=team_name,
                squad_market_value_eur=value,
                snapshot_date=pd.Timestamp(datetime.now(UTC).date()),
            )
    # Final fallback: any € value on the page.
    value = parse_market_value(html)
    if value is None:
        return None
    return TeamMarketValue(
        team_slug=team_slug,
        team_name=team_name,
        squad_market_value_eur=value,
        snapshot_date=pd.Timestamp(datetime.now(UTC).date()),
    )


def _make_session(
    cache_path: Path | None,
    expire_seconds: int,
) -> requests.Session:
    """Build a Session that caches GETs to SQLite if cache_path is given."""
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


def _load_robots(session: requests.Session) -> RobotFileParser:
    """Fetch + parse ``/robots.txt`` once per session.

    Failures (network error, malformed file) cause us to return a parser
    with no rules — i.e. fail-closed: refuse all subsequent requests. The
    polite default when robots.txt is unreachable is *not* to fetch.
    """
    parser = RobotFileParser()
    parser.set_url(f"{BASE_URL}/robots.txt")
    try:
        resp = session.get(f"{BASE_URL}/robots.txt", timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("could not fetch transfermarkt robots.txt: %s; refusing to scrape", exc)
        # Return a parser that disallows everything (empty parse_lines).
        parser.parse(["User-agent: *", "Disallow: /"])
        return parser
    parser.parse(resp.text.splitlines())
    return parser


def fetch_team_market_value(
    team_url: str,
    *,
    session: requests.Session | None = None,
    cache_path: Path | None = DEFAULT_CACHE,
    expire_seconds: int = 7 * 24 * 3600,
    robots: RobotFileParser | None = None,
) -> TeamMarketValue | None:
    """Fetch + parse one Transfermarkt team page.

    Respects ``robots.txt`` — refuses the request when the User-Agent is
    disallowed. Returns ``None`` on robots refusal, HTTP error, or parse
    failure, so the caller can keep iterating without try/except noise.
    """
    sess = session or _make_session(cache_path, expire_seconds)
    rp = robots or _load_robots(sess)
    if not rp.can_fetch(USER_AGENT, team_url):
        logger.warning("robots.txt disallows %s for our UA; skipping", team_url)
        return None
    try:
        resp = sess.get(team_url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("transfermarkt GET %s failed: %s", team_url, exc)
        return None
    # Derive the team slug from the URL — the second URL segment under /
    # is the readable name (e.g. "argentinien").
    path_parts = [p for p in urlparse(team_url).path.split("/") if p]
    team_slug = path_parts[0] if path_parts else "unknown"
    return parse_squad_page(resp.text, team_slug=team_slug)


def write_snapshot(
    rows: list[TeamMarketValue],
    target_dir: Path = DEFAULT_TARGET,
    *,
    today: datetime | None = None,
) -> Path:
    """Write a single dated parquet covering all scraped teams.

    Layout: ``data/raw/transfermarkt/squad_market_value_<YYYY-MM-DD>.parquet``.
    Idempotent within a day — overwriting today's snapshot is fine.
    """
    today = today or datetime.now(UTC)
    target_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "team_slug": r.team_slug,
                "team_name": r.team_name,
                "squad_market_value_eur": r.squad_market_value_eur,
                "snapshot_date": pd.Timestamp(today.date()),
            }
            for r in rows
        ]
    )
    out = target_dir / f"squad_market_value_{today:%Y-%m-%d}.parquet"
    df.to_parquet(out, index=False)
    return out


def load_latest_snapshot(target_dir: Path = DEFAULT_TARGET) -> pd.DataFrame:
    """Load the most recent ``squad_market_value_*.parquet`` snapshot."""
    paths = sorted(target_dir.glob("squad_market_value_*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"no transfermarkt snapshot in {target_dir}; "
            "run the manual transfermarkt_refresh job first."
        )
    return pd.read_parquet(paths[-1])


def fetch_squad_market_values(
    team_urls: dict[str, str],
    target_dir: Path = DEFAULT_TARGET,
    *,
    session: requests.Session | None = None,
    today: datetime | None = None,
) -> Path | None:
    """End-to-end: fetch every team in ``team_urls`` and write one parquet.

    ``team_urls`` maps the dashboard team name (e.g. ``"Argentina"``) to
    the canonical Transfermarkt URL. Mis-mapped or HTTP-failing entries
    are logged and skipped; if everything fails, no parquet is written
    and the function returns ``None``.
    """
    sess = session or _make_session(DEFAULT_CACHE, 7 * 24 * 3600)
    robots = _load_robots(sess)
    rows: list[TeamMarketValue] = []
    for display_name, url in team_urls.items():
        row = fetch_team_market_value(url, session=sess, robots=robots)
        if row is None:
            continue
        # Override the team name with the operator-supplied display name so
        # joins against the rest of the pipeline are exact.
        rows.append(
            TeamMarketValue(
                team_slug=row.team_slug,
                team_name=display_name,
                squad_market_value_eur=row.squad_market_value_eur,
                snapshot_date=row.snapshot_date,
            )
        )
    if not rows:
        logger.warning("transfermarkt: no rows scraped; not writing a parquet")
        return None
    return write_snapshot(rows, target_dir, today=today)


__all__ = [
    "BASE_URL",
    "DEFAULT_CACHE",
    "DEFAULT_TARGET",
    "TeamMarketValue",
    "USER_AGENT",
    "fetch_squad_market_values",
    "fetch_team_market_value",
    "load_latest_snapshot",
    "parse_market_value",
    "parse_squad_page",
    "write_snapshot",
]
