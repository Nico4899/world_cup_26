"""Unit tests for the FBref ingester."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from wc2026.ingest.fbref import (
    _strip_html_comments,
    fetch_team_match_logs,
    load_latest_snapshot,
    parse_match_log_html,
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"


def _fixture_html() -> str:
    return (FIXTURE_DIR / "fbref_match_log_sample.html").read_text(encoding="utf-8")


def test_strip_comments_removes_only_markers_not_contents() -> None:
    html = "<p>a</p><!-- <p>kept</p> --><p>b</p>"
    stripped = _strip_html_comments(html)
    assert "<p>kept</p>" in stripped
    assert "<!--" not in stripped
    assert "-->" not in stripped


def test_parse_match_log_extracts_three_real_matches() -> None:
    df = parse_match_log_html(_fixture_html(), team="Argentina")
    # Three real matches; the "Match Logs" summary row is filtered out.
    assert len(df) == 3
    assert list(df["opponent"]) == ["Bolivia", "Brazil", "Mexico"]


def test_parse_match_log_xg_columns_typed_as_float() -> None:
    df = parse_match_log_html(_fixture_html(), team="Argentina")
    assert df["xg_for"].dtype.kind == "f"
    assert df["xg_against"].dtype.kind == "f"
    assert df["xg_for"].tolist() == [2.7, 1.2, 1.9]
    assert df["xg_against"].tolist() == [0.4, 1.4, 0.8]


def test_parse_match_log_gf_ga_columns_typed_as_int64() -> None:
    df = parse_match_log_html(_fixture_html(), team="Argentina")
    assert str(df["gf"].dtype) == "Int64"
    assert df["gf"].tolist() == [3, 1, 2]


def test_parse_match_log_returns_empty_when_no_xg_table() -> None:
    df = parse_match_log_html(
        "<html><body><p>no relevant table</p></body></html>",
        team="X",
    )
    assert df.empty
    # Empty result still has the documented column set.
    assert set(df.columns) >= {"match_date", "xg_for", "xg_against"}


class _StubResponse:
    def __init__(self, text: str):
        self.text = text
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None


class _StubSession:
    def __init__(self, responses: dict[str, str]):
        self._responses = responses
        self.headers: dict[str, str] = {}
        self.calls: list[str] = []

    def get(self, url: str, **_):
        self.calls.append(url)
        return _StubResponse(self._responses[url])


def test_fetch_team_match_logs_writes_combined_parquet(tmp_path: Path) -> None:
    session = _StubSession({"https://fbref.test/argentina": _fixture_html()})
    out = fetch_team_match_logs(
        [("Argentina", "https://fbref.test/argentina")],
        session=session,
        target_dir=tmp_path,
        today=datetime(2026, 5, 23, tzinfo=UTC),
    )
    assert out.name == "fbref_xg_2026-05-23.parquet"
    df = pd.read_parquet(out)
    assert len(df) == 3
    assert (df["team"] == "Argentina").all()


def test_load_latest_snapshot_raises_when_empty(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        load_latest_snapshot(tmp_path)
