"""Unit tests for the openfootball cup.txt parser."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from wc2026.ingest.openfootball import (
    build_group_assignment,
    fetch_cup_txt,
    load_latest_assignment,
    parse_cup_txt,
    write_group_assignment_json,
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"


def _fixture_text() -> str:
    return (FIXTURE_DIR / "openfootball_cup_2026.txt").read_text(encoding="utf-8")


def test_parse_recovers_12_groups_of_4() -> None:
    groups = parse_cup_txt(_fixture_text())
    assert set(groups.keys()) == set("ABCDEFGHIJKL")
    assert all(len(members) == 4 for members in groups.values())


def test_parse_strips_three_letter_country_codes() -> None:
    groups = parse_cup_txt(_fixture_text())
    assert groups["A"][0] == "Mexico"
    assert groups["B"][0] == "Canada"
    assert groups["C"][0] == "United States"


def test_parse_ignores_matchday_section() -> None:
    groups = parse_cup_txt(_fixture_text())
    # Match-listing lines start without "N. " and must not be added as teams.
    assert all("@" not in t for members in groups.values() for t in members)


def test_parse_rejects_when_wrong_group_count() -> None:
    text = "= Header =\n\nGroup A:\n 1. T1\n 2. T2\n 3. T3\n 4. T4\n"
    with pytest.raises(ValueError, match=r"A\.\.L"):
        parse_cup_txt(text)


def test_parse_rejects_when_group_has_wrong_size() -> None:
    # Take the fixture and chop one line off Group A.
    text = _fixture_text().replace(" 4. Team A4\n", "")
    with pytest.raises(ValueError, match="must have 4 teams"):
        parse_cup_txt(text)


def test_parse_rejects_when_team_appears_twice() -> None:
    text = _fixture_text().replace(" 1. Team L1", " 1. Mexico")
    with pytest.raises(ValueError, match="duplicate"):
        parse_cup_txt(text)


def test_build_group_assignment_returns_dataclass_with_citation() -> None:
    assignment = build_group_assignment(_fixture_text(), citation="snapshot 2026-05-23")
    assert assignment.citation == "snapshot 2026-05-23"
    assert assignment.groups["A"] == ("Mexico", "Team A2", "Team A3", "Team A4")


def test_write_group_assignment_json_roundtrip(tmp_path: Path) -> None:
    assignment = build_group_assignment(_fixture_text())
    out = write_group_assignment_json(assignment, out_path=tmp_path / "assignment.json")
    payload: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    assert payload["source"].startswith("openfootball")
    assert payload["groups"]["A"] == ["Mexico", "Team A2", "Team A3", "Team A4"]


class _StubResponse:
    def __init__(self, text: str):
        self.text = text
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None


class _StubSession:
    def __init__(self, text: str):
        self._text = text
        self.headers: dict[str, str] = {}
        self.calls: list[str] = []

    def get(self, url: str, **_):
        self.calls.append(url)
        return _StubResponse(self._text)


def test_fetch_cup_txt_writes_dated_snapshot(tmp_path: Path) -> None:
    session = _StubSession(_fixture_text())
    out = fetch_cup_txt(
        url="https://example.test/cup.txt",
        session=session,
        target_dir=tmp_path,
        today=datetime(2026, 5, 23, tzinfo=UTC),
    )
    assert out.name == "cup_2026-05-23.txt"
    assert out.read_text(encoding="utf-8").startswith("= World Cup 2026 =")


def test_load_latest_assignment_returns_none_if_no_snapshot(tmp_path: Path) -> None:
    assert load_latest_assignment(tmp_path) is None


def test_load_latest_assignment_uses_most_recent_snapshot(tmp_path: Path) -> None:
    for d in ["2026-05-20", "2026-05-22", "2026-05-21"]:
        (tmp_path / f"cup_{d}.txt").write_text(_fixture_text(), encoding="utf-8")
    assignment = load_latest_assignment(tmp_path)
    assert assignment is not None
    assert "2026-05-22" in assignment.citation
