"""Unit tests for the Jürisoo Kaggle ingester.

Network/Kaggle-auth-dependent tests live under tests/integration/ and are skipped by
default; these unit tests use a synthetic CSV that mirrors the upstream schema.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wc2026.ingest.kaggle_intl import (
    RESULTS_COLUMNS,
    DatasetPaths,
    basic_stats,
    load_results,
)


def _write_synthetic_results(target: Path) -> Path:
    """Drop a 4-row synthetic results.csv at target/results.csv."""
    target.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "date": ["1872-11-30", "2018-07-15", "2022-12-18", "2026-06-11"],
            "home_team": ["Scotland", "France", "Argentina", "Mexico"],
            "away_team": ["England", "Croatia", "France", "Poland"],
            "home_score": [0, 4, 3, 2],
            "away_score": [0, 2, 3, 1],
            "tournament": [
                "Friendly",
                "FIFA World Cup",
                "FIFA World Cup",
                "FIFA World Cup",
            ],
            "city": ["Glasgow", "Moscow", "Lusail", "Mexico City"],
            "country": ["Scotland", "Russia", "Qatar", "Mexico"],
            "neutral": [False, True, True, False],
        }
    )
    csv = target / "results.csv"
    df.to_csv(csv, index=False)
    return csv


def test_dataset_paths_from_root(tmp_path: Path) -> None:
    paths = DatasetPaths.from_root(tmp_path)
    assert paths.root == tmp_path
    assert paths.results == tmp_path / "results.csv"
    assert paths.goalscorers == tmp_path / "goalscorers.csv"
    assert paths.shootouts == tmp_path / "shootouts.csv"


def test_load_results_happy_path(tmp_path: Path) -> None:
    _write_synthetic_results(tmp_path)
    df = load_results(tmp_path)
    assert len(df) == 4
    for col in RESULTS_COLUMNS:
        assert col in df.columns, f"missing {col}"
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["neutral"].dtype == bool
    assert df["home_score"].dtype == "Int64"
    assert df.loc[3, "home_team"] == "Mexico"
    assert df.loc[3, "date"].year == 2026


def test_load_results_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"results\.csv"):
        load_results(tmp_path)


def test_load_results_missing_column(tmp_path: Path) -> None:
    (tmp_path / "results.csv").write_text("date,home_team\n2026-01-01,USA\n")
    with pytest.raises(ValueError, match="missing expected columns"):
        load_results(tmp_path)


def test_basic_stats(tmp_path: Path) -> None:
    _write_synthetic_results(tmp_path)
    stats = basic_stats(load_results(tmp_path))
    assert stats["n_matches"] == 4
    assert stats["date_min"] == "1872-11-30"
    assert stats["date_max"] == "2026-06-11"
    # France appears in both row 1 (home) and row 2 (away) → 7 unique teams across 4 matches.
    assert stats["n_teams"] == 7
    assert stats["n_tournaments"] == 2
    assert stats["neutral_pct"] == 50.0


def test_basic_stats_empty() -> None:
    empty = pd.DataFrame(
        {
            "date": pd.to_datetime([]),
            "home_team": pd.Series([], dtype="string"),
            "away_team": pd.Series([], dtype="string"),
            "home_score": pd.Series([], dtype="Int64"),
            "away_score": pd.Series([], dtype="Int64"),
            "tournament": pd.Series([], dtype="string"),
            "city": pd.Series([], dtype="string"),
            "country": pd.Series([], dtype="string"),
            "neutral": pd.Series([], dtype=bool),
        }
    )
    stats = basic_stats(empty)
    assert stats["n_matches"] == 0
    assert stats["date_min"] is None
    assert stats["date_max"] is None
    assert stats["neutral_pct"] == 0.0
