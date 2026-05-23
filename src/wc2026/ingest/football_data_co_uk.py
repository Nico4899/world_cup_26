"""Ingest historical match results + bookmaker closing odds from football-data.co.uk.

Source: https://www.football-data.co.uk

The site publishes one CSV per (league, season) under
``https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv`` —
e.g. ``mmz4281/2425/E0.csv`` for the English Premier League 2024-25. Closing
odds for several bookmakers (Bet365 / Pinnacle / Betfair / William Hill / …)
are included; we keep the canonical Bet365 closing 1X2 (``B365CH/D/A``) and
fall back to Pinnacle closing (``PCH/D/A``) when Bet365 is unavailable.

What this is for
----------------
The blueprint cited football-data.co.uk as a benchmark source for **World
Cup** bookmaker odds — on inspection the site has **no WC odds**, only club
competitions. We still ingest it because club-level closing odds are the
gold-standard reference for *xG-model* calibration: a well-fit xG model on a
domestic league should produce 1X2 probabilities whose log-loss matches the
bookmaker market. See Phase 3's hindcast gate.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import requests_cache

BASE_URL = "https://www.football-data.co.uk/mmz4281"

DEFAULT_TARGET = Path("data/raw/football_data_co_uk")
DEFAULT_CACHE = Path("data/raw/football_data_co_uk/.http_cache")
DEFAULT_CACHE_EXPIRY_SECONDS = 24 * 3600

USER_AGENT = (
    "wc2026-predictor/0.1 "
    "(+https://github.com/Nico4899/world_cup_26; nico.fliegel@gmail.com) "
    "personal-research; calibrated WC 2026 predictions"
)

# Curated season+league pairs used as the xG calibration corpus. Override via
# `fetch_seasons` argument if you want a wider or narrower set.
# Season codes are YYYY/YY-YY (e.g. "2425" = 2024-25); league codes per the
# football-data.co.uk notes.txt (E0=Premier League, SP1=La Liga, D1=Bundesliga,
# I1=Serie A, F1=Ligue 1).
DEFAULT_CALIBRATION_SET: tuple[tuple[str, str], ...] = (
    ("2425", "E0"),
    ("2324", "E0"),
    ("2425", "SP1"),
    ("2324", "SP1"),
    ("2425", "D1"),
    ("2324", "D1"),
)

logger = logging.getLogger(__name__)


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
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/csv,*/*"})
    return session


def parse_csv(csv_text: str) -> pd.DataFrame:
    """Parse a football-data.co.uk league CSV into the canonical schema.

    Output columns:
        match_date (datetime64[ns]), home_team (str), away_team (str),
        fthg (Int64), ftag (Int64), ftr (str — "H" / "D" / "A"),
        odds_home (float64), odds_draw (float64), odds_away (float64),
        odds_source (str — "B365C" / "PC" / "" if neither present)

    Rows missing date or both bookmaker pairs are dropped.
    """
    try:
        df = pd.read_csv(StringIO(csv_text), encoding="latin-1", on_bad_lines="skip")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    df.columns = [str(c).strip() for c in df.columns]
    if "Date" not in df.columns or "HomeTeam" not in df.columns:
        return pd.DataFrame()
    # football-data.co.uk publishes dates as DD/MM/YY (2-digit year). Try the
    # documented format first, then fall back to ISO for any oddly-formatted row.
    parsed_date = pd.to_datetime(df["Date"], format="%d/%m/%y", errors="coerce")
    iso_fallback = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    parsed_date = parsed_date.fillna(iso_fallback)
    out = pd.DataFrame(
        {
            "match_date": parsed_date,
            "home_team": df["HomeTeam"].astype(str).str.strip(),
            "away_team": df["AwayTeam"].astype(str).str.strip(),
            "fthg": pd.to_numeric(df.get("FTHG"), errors="coerce").astype("Int64"),
            "ftag": pd.to_numeric(df.get("FTAG"), errors="coerce").astype("Int64"),
            "ftr": df.get("FTR", "").astype(str).str.strip().str.upper(),
        }
    )
    # Bookmaker odds: prefer Bet365 closing, fall back to Pinnacle closing.
    odds_home = pd.to_numeric(df.get("B365CH"), errors="coerce")
    odds_draw = pd.to_numeric(df.get("B365CD"), errors="coerce")
    odds_away = pd.to_numeric(df.get("B365CA"), errors="coerce")
    source = pd.Series(["B365C"] * len(df))
    pc_home = pd.to_numeric(df.get("PCH"), errors="coerce")
    pc_draw = pd.to_numeric(df.get("PCD"), errors="coerce")
    pc_away = pd.to_numeric(df.get("PCA"), errors="coerce")
    needs_pinnacle = odds_home.isna() | odds_draw.isna() | odds_away.isna()
    odds_home = odds_home.where(~needs_pinnacle, pc_home)
    odds_draw = odds_draw.where(~needs_pinnacle, pc_draw)
    odds_away = odds_away.where(~needs_pinnacle, pc_away)
    source = source.where(~needs_pinnacle, "PC")
    no_odds = odds_home.isna() | odds_draw.isna() | odds_away.isna()
    source = source.where(~no_odds, "")
    out["odds_home"] = odds_home
    out["odds_draw"] = odds_draw
    out["odds_away"] = odds_away
    out["odds_source"] = source.astype(str)
    out = out.dropna(subset=["match_date"]).reset_index(drop=True)
    return out


def implied_probabilities(odds: pd.DataFrame) -> pd.DataFrame:
    """Convert decimal closing odds → normalised implied probabilities.

    Adds columns ``p_home``, ``p_draw``, ``p_away`` summing to 1 per row. Rows
    with missing odds get NaN for all three; the overround (bookmaker margin)
    is silently removed by the normalisation.
    """
    if odds.empty:
        return odds.assign(p_home=[], p_draw=[], p_away=[])
    raw_h = 1.0 / odds["odds_home"]
    raw_d = 1.0 / odds["odds_draw"]
    raw_a = 1.0 / odds["odds_away"]
    total = raw_h + raw_d + raw_a
    out = odds.copy()
    out["p_home"] = raw_h / total
    out["p_draw"] = raw_d / total
    out["p_away"] = raw_a / total
    return out


def fetch_league_csv(
    season_code: str,
    league_code: str,
    *,
    session: requests.Session | None = None,
    target_dir: Path = DEFAULT_TARGET,
    today: datetime | None = None,
) -> Path:
    """Download one (season, league) CSV and write a Parquet snapshot.

    Returns the Parquet path. The on-disk filename is
    ``{season}_{league}_<YYYY-MM-DD>.parquet`` so we can keep multiple snapshots.
    """
    today = today or datetime.now(UTC)
    target_dir.mkdir(parents=True, exist_ok=True)
    sess = session or _make_session()
    url = f"{BASE_URL}/{season_code}/{league_code}.csv"
    resp = sess.get(url, timeout=30)
    resp.raise_for_status()
    df = parse_csv(resp.text)
    out = target_dir / f"{season_code}_{league_code}_{today:%Y-%m-%d}.parquet"
    df.to_parquet(out, index=False)
    return out


def fetch_calibration_corpus(
    pairs: list[tuple[str, str]] | None = None,
    *,
    session: requests.Session | None = None,
    target_dir: Path = DEFAULT_TARGET,
    today: datetime | None = None,
) -> list[Path]:
    """Fetch every (season, league) CSV in ``pairs``. Returns the Parquet paths."""
    sess = session or _make_session()
    target = pairs if pairs is not None else list(DEFAULT_CALIBRATION_SET)
    paths: list[Path] = []
    for season_code, league_code in target:
        try:
            paths.append(
                fetch_league_csv(
                    season_code,
                    league_code,
                    session=sess,
                    target_dir=target_dir,
                    today=today,
                )
            )
        except requests.HTTPError as exc:
            logger.warning("football-data.co.uk %s/%s failed: %s", season_code, league_code, exc)
    return paths


def load_latest_for_league(
    season_code: str,
    league_code: str,
    target_dir: Path = DEFAULT_TARGET,
) -> pd.DataFrame:
    paths = sorted(target_dir.glob(f"{season_code}_{league_code}_*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No {season_code}_{league_code}_*.parquet snapshots in {target_dir}"
        )
    return pd.read_parquet(paths[-1])


__all__ = [
    "BASE_URL",
    "DEFAULT_CACHE",
    "DEFAULT_CALIBRATION_SET",
    "DEFAULT_TARGET",
    "fetch_calibration_corpus",
    "fetch_league_csv",
    "implied_probabilities",
    "load_latest_for_league",
    "parse_csv",
]
