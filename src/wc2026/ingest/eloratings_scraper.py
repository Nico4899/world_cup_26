"""Polite snapshot of World Football Elo Ratings (eloratings.net).

Approach
--------
The eloratings.net front page is a JS SPA — there is no scrapable HTML table.
The same code that renders the page loads two TSV data files from the server.
We fetch those directly, which is exactly two HTTP requests per refresh:

    https://www.eloratings.net/World.tsv       (~30 KB, 244 rows, 31 cols)
    https://www.eloratings.net/en.teams.tsv    (~7  KB, ~331 rows, 2 cols)

We send a clear User-Agent that names the project, links the repo, and gives a
contact email so the site maintainer can reach out if anything is wrong. Results
are cached via requests-cache (SQLite-backed, survives process restarts) with a
1-hour default expiry — the upstream files only change after international matches.

Site licence note: eloratings.net permits non-commercial use; this project is
strictly personal/educational. Cite "World Football Elo Ratings, eloratings.net"
in any user-facing surface that displays these numbers.

Column schema (decoded from the site's `scripts/ratings.js` grid definitions):
see WORLD_TSV_COLUMNS below. Negative values use the Unicode minus sign
(U+2212) rather than ASCII hyphen-minus, and must be normalised on parse.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests
import requests_cache

BASE_URL = "https://www.eloratings.net"
WORLD_TSV_URL = f"{BASE_URL}/World.tsv"
TEAMS_TSV_URL = f"{BASE_URL}/en.teams.tsv"

USER_AGENT = (
    "wc2026-predictor/0.1 "
    "(+https://github.com/Nico4899/world_cup_26; nico.fliegel@gmail.com) "
    "personal-research; calibrated WC 2026 predictions"
)

DEFAULT_TARGET = Path("data/raw/elo")
DEFAULT_CACHE = Path("data/raw/elo/.http_cache")  # requests-cache appends .sqlite

WORLD_TSV_COLUMNS: tuple[str, ...] = (
    "local_rank",
    "global_rank",
    "code",
    "rating",
    "rank_max",
    "rating_max",
    "rank_avg",
    "rating_avg",
    "rank_min",
    "rating_min",
    "rank_3m_change",
    "rating_3m_change",
    "rank_6m_change",
    "rating_6m_change",
    "rank_1y_change",
    "rating_1y_change",
    "rank_2y_change",
    "rating_2y_change",
    "rank_5y_change",
    "rating_5y_change",
    "rank_10y_change",
    "rating_10y_change",
    "matches_total",
    "matches_home",
    "matches_away",
    "matches_neutral",
    "wins",
    "losses",
    "draws",
    "goals_for",
    "goals_against",
)
_INT_COLUMNS: tuple[str, ...] = tuple(c for c in WORLD_TSV_COLUMNS if c != "code")

UNICODE_MINUS = "−"  # noqa: RUF001 — this constant exists precisely to name the char


def _normalise_minus(text: str) -> str:
    """Replace Unicode minus (U+2212) with ASCII hyphen-minus so numeric parse works."""
    return text.replace(UNICODE_MINUS, "-")


def parse_world_tsv(text: str) -> pd.DataFrame:
    """Parse the raw World.tsv text into a typed DataFrame.

    All numeric columns are Int64 (nullable). The `code` column is string.
    Tiny teams (e.g. Marshall Islands, 2 international matches ever) emit a lone
    Unicode minus in change columns where no history is available; those become NaN.

    Raises ValueError on unexpected column count.
    """
    text = _normalise_minus(text)
    first_line = next((ln for ln in text.splitlines() if ln.strip()), "")
    n_cols = first_line.count("\t") + 1
    if n_cols != len(WORLD_TSV_COLUMNS):
        raise ValueError(f"World.tsv has {n_cols} columns, expected {len(WORLD_TSV_COLUMNS)}")
    df = pd.read_csv(
        io.StringIO(text),
        sep="\t",
        header=None,
        names=list(WORLD_TSV_COLUMNS),
        engine="c",
        dtype={"code": "string"},
    )
    for col in _INT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def parse_teams_tsv(text: str) -> pd.DataFrame:
    """Parse en.teams.tsv into a (code, team_name) DataFrame.

    The upstream file is variable-width: each row is the team code followed by 1+
    progressively shorter display variants of the name (e.g. "Antigua and Barbuda" |
    "Antigua & Barbuda" | "Antigua/Barbuda" | …). We keep the first (longest) variant.

    Locative-phrase rows ("BS_loc", "in the Bahamas") are filtered out — they're UI
    prepositional strings, not team identifiers.
    """
    rows: list[tuple[str, str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        first_tab = line.find("\t")
        if first_tab < 0:
            continue
        code = line[:first_tab]
        if code.endswith("_loc"):
            continue
        rest = line[first_tab + 1 :]
        # second-and-later tab-separated tokens are shorter abbreviations; take only the longest.
        second_tab = rest.find("\t")
        team_name = rest if second_tab < 0 else rest[:second_tab]
        rows.append((code, team_name))
    return pd.DataFrame(rows, columns=["code", "team_name"]).astype(
        {"code": "string", "team_name": "string"}
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
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/plain,*/*"})
    return session


def fetch_current_ratings(
    target_dir: Path = DEFAULT_TARGET,
    *,
    cache_path: Path | None = DEFAULT_CACHE,
    expire_seconds: int = 3600,
    today: datetime | None = None,
    session: requests.Session | None = None,
) -> Path:
    """Fetch World.tsv + en.teams.tsv, join, and write a dated Parquet snapshot.

    Returns the path to the new Parquet file at
    ``target_dir/elo_current_<YYYY-MM-DD>.parquet``. Idempotent within a day:
    the destination file is overwritten so the snapshot for today always reflects
    the latest data.
    """
    today = today or datetime.now(UTC)
    target_dir.mkdir(parents=True, exist_ok=True)
    sess = session or _make_session(cache_path, expire_seconds)

    world_resp = sess.get(WORLD_TSV_URL, timeout=30)
    world_resp.raise_for_status()
    world_df = parse_world_tsv(world_resp.text)

    teams_resp = sess.get(TEAMS_TSV_URL, timeout=30)
    teams_resp.raise_for_status()
    teams_df = parse_teams_tsv(teams_resp.text)

    df = world_df.merge(teams_df, on="code", how="left")
    df["snapshot_date"] = pd.to_datetime(today.date())

    out = target_dir / f"elo_current_{today:%Y-%m-%d}.parquet"
    df.to_parquet(out, index=False)
    return out


def load_latest_snapshot(target_dir: Path = DEFAULT_TARGET) -> pd.DataFrame:
    """Load the most recent ``elo_current_*.parquet`` snapshot in target_dir."""
    paths = sorted(target_dir.glob("elo_current_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No elo_current_*.parquet snapshots in {target_dir}")
    return pd.read_parquet(paths[-1])
