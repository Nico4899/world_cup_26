"""Unit tests for the football-data.org ingester (no network).

The parser tests run against ``tests/fixtures/football_data_org_sample.json`` —
a hand-crafted minimal payload matching the v4 /competitions/{code}/matches
schema. Rate-limiter tests exercise the sliding-window logic deterministically.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from wc2026.ingest import football_data_org as fdo

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "football_data_org_sample.json"


def test_fixture_file_exists():
    assert FIXTURE.exists(), "fixture is required for offline tests"


def test_parse_competition_matches_returns_typed_dataframe():
    payload = fdo.load_fixture(FIXTURE)
    df = fdo.parse_competition_matches(payload)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert set(df.columns) == {
        "match_id",
        "utc_date",
        "status",
        "matchday",
        "stage",
        "group",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "winner",
        "venue",
    }
    assert df["match_id"].dtype == "Int64"
    assert df["home_score"].dtype == "Int64"
    assert df["utc_date"].dt.tz is not None


def test_parser_handles_played_and_scheduled_rows_correctly():
    df = fdo.parse_competition_matches(fdo.load_fixture(FIXTURE))

    played = df[df["status"] == "FINISHED"].iloc[0]
    assert played["home_team"] == "Mexico"
    assert played["away_team"] == "Senegal"
    assert int(played["home_score"]) == 2
    assert int(played["away_score"]) == 1

    scheduled = df[df["status"] == "SCHEDULED"].iloc[0]
    assert pd.isna(scheduled["home_score"])
    assert pd.isna(scheduled["away_score"])
    assert scheduled["group"] == "GROUP_A"


def test_parser_handles_placeholder_team_names_as_na():
    """The Final stage row in the fixture has null team names (bracket TBD)."""
    df = fdo.parse_competition_matches(fdo.load_fixture(FIXTURE))
    final_row = df[df["stage"] == "FINAL"].iloc[0]
    assert pd.isna(final_row["home_team"])
    assert pd.isna(final_row["away_team"])


def test_parser_empty_payload_returns_empty_dataframe():
    assert fdo.parse_competition_matches({"matches": []}).empty
    assert fdo.parse_competition_matches({}).empty


def test_fetch_competition_matches_requires_api_key(monkeypatch):
    monkeypatch.delenv(fdo.ENV_API_KEY, raising=False)
    with pytest.raises(fdo.MissingApiKeyError):
        fdo.fetch_competition_matches()


def test_fetch_match_requires_api_key(monkeypatch):
    monkeypatch.delenv(fdo.ENV_API_KEY, raising=False)
    with pytest.raises(fdo.MissingApiKeyError):
        fdo.fetch_match(123)


def test_rate_limiter_throttles_when_window_full():
    """11th acquire in a 1-second window should sleep until the window slides."""
    limiter = fdo._RateLimiter.make(limit=10, window=1.0)
    sleeps: list[float] = []

    def fake_sleep(duration: float) -> None:
        sleeps.append(duration)

    base = time.monotonic()
    for i in range(10):
        limiter.acquire(now=base + i * 0.01, sleep=fake_sleep)
    assert sleeps == []

    limiter.acquire(now=base + 0.5, sleep=fake_sleep)
    assert len(sleeps) == 1
    assert sleeps[0] > 0


def test_rate_limiter_does_not_throttle_when_window_clears():
    limiter = fdo._RateLimiter.make(limit=2, window=1.0)
    sleeps: list[float] = []
    base = time.monotonic()
    limiter.acquire(now=base, sleep=sleeps.append)
    limiter.acquire(now=base + 0.1, sleep=sleeps.append)
    limiter.acquire(now=base + 2.0, sleep=sleeps.append)
    assert sleeps == []
