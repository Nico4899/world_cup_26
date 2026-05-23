"""Unit tests for the eloratings.net scraper.

Uses real (small) TSV files captured once from the site, kept under tests/fixtures/.
No network calls in unit tests; the live integration test lives separately and is
skipped unless explicitly selected.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wc2026.ingest.eloratings_scraper import (
    UNICODE_MINUS,
    WORLD_TSV_COLUMNS,
    _normalise_minus,
    load_latest_snapshot,
    parse_teams_tsv,
    parse_world_tsv,
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"


def test_normalise_minus_replaces_unicode_with_ascii() -> None:
    assert _normalise_minus(f"foo{UNICODE_MINUS}1") == "foo-1"
    assert _normalise_minus("no minus here") == "no minus here"


def test_parse_world_tsv_real_fixture() -> None:
    text = (FIXTURE_DIR / "World.tsv").read_text(encoding="utf-8")
    df = parse_world_tsv(text)
    assert len(df) >= 200
    assert list(df.columns) == list(WORLD_TSV_COLUMNS)
    # types
    assert df["code"].dtype == "string"
    for col in WORLD_TSV_COLUMNS:
        if col == "code":
            continue
        assert df[col].dtype == "Int64", f"{col} should be Int64, got {df[col].dtype}"
    # spot-check #1
    top = df.iloc[0]
    assert top["local_rank"] == 1
    assert top["global_rank"] == 1
    assert top["code"] == "ES"
    assert 1900 <= top["rating"] <= 2300
    # match-count sanity: home + away + neutral == total
    assert (df["matches_home"] + df["matches_away"] + df["matches_neutral"]).equals(
        df["matches_total"]
    )
    # win + loss + draw == total
    assert (df["wins"] + df["losses"] + df["draws"]).equals(df["matches_total"])


def test_parse_world_tsv_handles_unicode_minus() -> None:
    # synthetic 1-row TSV with the Unicode minus sign
    fields: list[str] = [
        "1",
        "1",
        "XX",
        "1500",  # local_rank, global_rank, code, rating
        "10",
        "1700",  # rank_max, rating_max
        "100",
        "1450",  # rank_avg, rating_avg
        "200",
        "1100",  # rank_min, rating_min
    ]
    # 6 change-pairs, all negative (3m, 6m, 1y, 2y, 5y, 10y)
    fields.extend([f"{UNICODE_MINUS}5", f"{UNICODE_MINUS}50"] * 6)
    fields.extend(["100", "40", "40", "20", "30", "30", "40", "100", "120"])
    text = "\t".join(fields) + "\n"
    df = parse_world_tsv(text)
    row = df.iloc[0]
    assert row["rating_3m_change"] == -50
    assert row["rank_3m_change"] == -5
    assert row["rating_10y_change"] == -50


def test_parse_world_tsv_wrong_column_count_raises() -> None:
    with pytest.raises(ValueError, match="columns, expected"):
        parse_world_tsv("1\t2\t3\n")


def test_parse_teams_tsv_real_fixture() -> None:
    text = (FIXTURE_DIR / "en.teams.tsv").read_text(encoding="utf-8")
    df = parse_teams_tsv(text)
    assert len(df) >= 200
    assert list(df.columns) == ["code", "team_name"]
    spain = df[df["code"] == "ES"].iloc[0]
    assert spain["team_name"] == "Spain"


def test_load_latest_snapshot_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_latest_snapshot(tmp_path)


def test_load_latest_snapshot_picks_newest(tmp_path: Path) -> None:
    pd.DataFrame({"code": ["ES"]}).to_parquet(tmp_path / "elo_current_2024-01-01.parquet")
    pd.DataFrame({"code": ["AR"]}).to_parquet(tmp_path / "elo_current_2026-05-23.parquet")
    df = load_latest_snapshot(tmp_path)
    assert df.iloc[0]["code"] == "AR"


def test_fetch_current_ratings_with_mock_session(tmp_path: Path) -> None:
    """End-to-end test of fetch_current_ratings using a stubbed Session (no network)."""

    class _StubResponse:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return None

    class _StubSession:
        def __init__(self, world: str, teams: str) -> None:
            self._world = world
            self._teams = teams
            self.headers: dict[str, str] = {}

        def get(self, url: str, **_: object) -> _StubResponse:
            return _StubResponse(self._world if url.endswith("World.tsv") else self._teams)

    world_text = (FIXTURE_DIR / "World.tsv").read_text(encoding="utf-8")
    teams_text = (FIXTURE_DIR / "en.teams.tsv").read_text(encoding="utf-8")
    from wc2026.ingest.eloratings_scraper import fetch_current_ratings

    out = fetch_current_ratings(
        target_dir=tmp_path,
        cache_path=None,
        session=_StubSession(world_text, teams_text),  # type: ignore[arg-type]
    )
    assert out.exists()
    assert out.name.startswith("elo_current_")
    assert out.suffix == ".parquet"
    df = pd.read_parquet(out)
    assert len(df) >= 200
    assert "team_name" in df.columns  # came from the merge
    assert "rating" in df.columns
    spain = df[df["code"] == "ES"].iloc[0]
    assert spain["team_name"] == "Spain"
    assert spain["local_rank"] == 1
