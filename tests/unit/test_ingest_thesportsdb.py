"""Unit tests for the TheSportsDB ingester.

Network is stubbed via a minimal ``requests.Session`` lookalike; tests run on
captured JSON fixtures so no real HTTP call is made.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from wc2026.ingest.thesportsdb import (
    DEFAULT_ALIASES,
    fetch_team,
    fetch_team_assets,
    load_latest_snapshot,
    parse_team_lookup_response,
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"


def _load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


class _StubResponse:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload
        self.status_code = 200

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _StubSession:
    """Returns a payload keyed off the ``t`` query parameter."""

    def __init__(self, payloads: dict[str, dict[str, Any]]):
        self._payloads = payloads
        self.headers: dict[str, str] = {}
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def get(self, url: str, params: dict[str, Any] | None = None, **_):
        self.calls.append((url, params))
        key = (params or {}).get("t", "")
        return _StubResponse(self._payloads.get(key, {"teams": None}))


def test_parse_picks_first_soccer_entry_exact_match() -> None:
    payload = _load_fixture("thesportsdb_argentina.json")
    row = parse_team_lookup_response(payload, canonical_name="Argentina")
    assert row is not None
    assert row["team"] == "Argentina"
    assert row["thesportsdb_id"] == 133602
    assert row["crest_url"] == "https://example.test/badges/argentina.png"
    assert row["kit_home_color"] == "#75AADB"
    assert row["kit_away_color"] == "#000080"
    assert row["stadium_capacity"] == 83214
    assert row["stadium_city"] == "Buenos Aires"


def test_parse_returns_none_for_empty_payload() -> None:
    assert (
        parse_team_lookup_response(_load_fixture("thesportsdb_empty.json"), canonical_name="X")
        is None
    )
    assert parse_team_lookup_response({}, canonical_name="X") is None
    assert parse_team_lookup_response(None, canonical_name="X") is None


def test_parse_filters_out_non_soccer_entries() -> None:
    payload = {"teams": [{"idTeam": "1", "strTeam": "X", "strSport": "Baseball"}]}
    assert parse_team_lookup_response(payload, canonical_name="X") is None


def test_parse_prefers_exact_strteam_match_over_first_soccer() -> None:
    payload = {
        "teams": [
            {"idTeam": "1", "strTeam": "Argentina U20", "strSport": "Soccer"},
            {"idTeam": "2", "strTeam": "Argentina", "strSport": "Soccer"},
        ]
    }
    row = parse_team_lookup_response(payload, canonical_name="Argentina")
    assert row is not None
    assert row["thesportsdb_id"] == 2


def test_default_aliases_cover_known_us_korea_spellings() -> None:
    assert DEFAULT_ALIASES["United States"] == "USA"
    assert DEFAULT_ALIASES["South Korea"] == "Korea Republic"


def test_fetch_team_uses_alias_map_for_query_param() -> None:
    session = _StubSession(
        {"USA": _load_fixture("thesportsdb_argentina.json")}  # any payload to confirm the URL fired
    )
    fetch_team("United States", api_key="K", session=session)
    assert session.calls, "expected at least one GET"
    url, params = session.calls[0]
    assert "/searchteams.php" in url
    assert (params or {}).get("t") == "USA"


def test_fetch_team_passes_canonical_name_through_to_row() -> None:
    """Even though we query upstream as 'USA', the stored team is 'United States'."""
    session = _StubSession({"USA": _load_fixture("thesportsdb_argentina.json")})
    row = fetch_team("United States", api_key="K", session=session)
    assert row is not None
    assert row["team"] == "United States"


def test_fetch_team_assets_writes_dated_parquet(tmp_path) -> None:
    payload = _load_fixture("thesportsdb_argentina.json")
    session = _StubSession({"Argentina": payload, "Brazil": payload})
    out = fetch_team_assets(
        ["Argentina", "Brazil"],
        api_key="K",
        target_dir=tmp_path,
        session=session,
        today=datetime(2026, 5, 23, tzinfo=UTC),
    )
    assert out.name == "teams_2026-05-23.parquet"
    df = pd.read_parquet(out)
    assert list(df["team"]) == ["Argentina", "Brazil"]
    assert df["stadium_capacity"].tolist() == [83214, 83214]


def test_fetch_team_assets_skips_missing_teams_and_writes_remaining(tmp_path) -> None:
    session = _StubSession({"Argentina": _load_fixture("thesportsdb_argentina.json")})
    out = fetch_team_assets(
        ["Argentina", "Nowhereistan"],
        api_key="K",
        target_dir=tmp_path,
        session=session,
        today=datetime(2026, 5, 23, tzinfo=UTC),
    )
    df = pd.read_parquet(out)
    assert df["team"].tolist() == ["Argentina"]


def test_load_latest_snapshot_picks_most_recent_file(tmp_path) -> None:
    for d in ["2026-05-20", "2026-05-22", "2026-05-21"]:
        pd.DataFrame({"team": ["X"], "snapshot": [d]}).to_parquet(tmp_path / f"teams_{d}.parquet")
    df = load_latest_snapshot(tmp_path)
    assert df["snapshot"].iloc[0] == "2026-05-22"


def test_load_latest_snapshot_raises_when_directory_empty(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_latest_snapshot(tmp_path)
