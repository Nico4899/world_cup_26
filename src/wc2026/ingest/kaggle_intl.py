"""Ingest Mart Jürisoo's 'International football results' Kaggle dataset.

Source: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
Licence: CC0 (per Kaggle dataset page).

The dataset ships three CSVs:
  - results.csv     — one row per match (date, teams, score, tournament, venue, neutral flag)
  - goalscorers.csv — one row per goal (date, teams, scorer, minute, own_goal, penalty)
  - shootouts.csv   — one row per shootout (date, teams, winner)

Only `results.csv` is required for the bivariate Poisson fit. The other two will be
loaded later for the shootout submodel (Stage 0.x) and for player-level enrichment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATASET_SLUG = "martj42/international-football-results-from-1872-to-2017"
DEFAULT_TARGET = Path("data/raw/jurisoo")

RESULTS_COLUMNS: tuple[str, ...] = (
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
)


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    results: Path
    goalscorers: Path
    shootouts: Path

    @classmethod
    def from_root(cls, root: Path) -> DatasetPaths:
        return cls(
            root=root,
            results=root / "results.csv",
            goalscorers=root / "goalscorers.csv",
            shootouts=root / "shootouts.csv",
        )


def download_dataset(target_dir: Path = DEFAULT_TARGET, *, force: bool = False) -> DatasetPaths:
    """Download the Kaggle dataset into target_dir/ and unzip in place.

    Idempotent: if results.csv already exists in target_dir, no network call is made
    unless force=True. The Kaggle client is imported lazily so the rest of the module
    can be used (loaders, tests) without `kaggle` credentials configured.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    paths = DatasetPaths.from_root(target_dir)
    if paths.results.exists() and not force:
        return paths

    # Lazy import: keeps the module importable in environments without Kaggle creds.
    from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(DATASET_SLUG, path=str(target_dir), unzip=True, quiet=False)
    if not paths.results.exists():
        raise RuntimeError(
            f"Download finished but {paths.results} is missing — check Kaggle slug and contents."
        )
    return paths


def load_results(target_dir: Path = DEFAULT_TARGET) -> pd.DataFrame:
    """Load results.csv with typed columns, parsed dates, and validated schema.

    Returns a DataFrame indexed by row order with these dtypes:
      date         datetime64[ns]
      home_team    string
      away_team    string
      home_score   Int64   (nullable; some old fixtures lack a score)
      away_score   Int64
      tournament   string
      city         string
      country      string
      neutral      bool
    """
    paths = DatasetPaths.from_root(target_dir)
    if not paths.results.exists():
        raise FileNotFoundError(
            f"{paths.results} not found. Run download_dataset() first, or check the path."
        )
    df = pd.read_csv(
        paths.results,
        dtype={
            "home_team": "string",
            "away_team": "string",
            "home_score": "Int64",
            "away_score": "Int64",
            "tournament": "string",
            "city": "string",
            "country": "string",
        },
        parse_dates=["date"],
    )
    missing = set(RESULTS_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"results.csv is missing expected columns: {sorted(missing)}. "
            f"Got columns: {list(df.columns)}"
        )
    if df["neutral"].dtype != bool:
        df["neutral"] = df["neutral"].astype(bool)
    return df


def load_played(target_dir: Path = DEFAULT_TARGET) -> pd.DataFrame:
    """Return only matches with both scores recorded (i.e. actually played).

    The upstream CSV mixes played matches with pre-listed future fixtures (notably the
    WC 2026 group-stage draw, where venues are known but scores are NULL). Almost all
    training/feature code wants only the played subset.
    """
    df = load_results(target_dir)
    return df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)


def load_scheduled(target_dir: Path = DEFAULT_TARGET) -> pd.DataFrame:
    """Return only matches with NULL scores (pre-listed future fixtures).

    For WC 2026 this is the 72 group-stage matches with venues filled in; knockout
    matches cannot be pre-listed because opponents depend on group outcomes.
    """
    df = load_results(target_dir)
    return df[df["home_score"].isna() | df["away_score"].isna()].reset_index(drop=True)


def basic_stats(df: pd.DataFrame) -> dict[str, object]:
    """Tiny summary used by the CLI/smoke checks; not a test target."""
    return {
        "n_matches": len(df),
        "date_min": df["date"].min().date().isoformat() if len(df) else None,
        "date_max": df["date"].max().date().isoformat() if len(df) else None,
        "n_teams": int(pd.concat([df["home_team"], df["away_team"]]).nunique()),
        "n_tournaments": int(df["tournament"].nunique()),
        "neutral_pct": round(100 * df["neutral"].mean(), 2) if len(df) else 0.0,
    }
