"""Unit tests for the Wikipedia ingester (squads + FIFA Men's Ranking)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from wc2026.ingest.wikipedia import (
    fetch_all_squads,
    fetch_fifa_ranking,
    parse_fifa_ranking_html,
    parse_squad_wikitext,
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"


def _squad_wikitext() -> str:
    return (FIXTURE_DIR / "wikipedia_squad_sample.txt").read_text(encoding="utf-8")


def _ranking_html() -> str:
    return (FIXTURE_DIR / "wikipedia_fifa_ranking_sample.html").read_text(encoding="utf-8")


def test_squad_parser_extracts_three_players() -> None:
    df = parse_squad_wikitext(
        _squad_wikitext(),
        team="Argentina",
        tournament="FIFA World Cup 2026",
        snapshot_date=date(2026, 5, 23),
    )
    assert len(df) == 3
    assert set(df["player_name"]) == {
        "[[Emiliano Martínez]]",
        "[[Lionel Messi]]",
        "[[Lautaro Martínez]]",
    }


def test_squad_parser_extracts_positions_and_numbers() -> None:
    df = parse_squad_wikitext(
        _squad_wikitext(),
        team="Argentina",
        tournament="FIFA World Cup 2026",
        snapshot_date=date(2026, 5, 23),
    )
    by_no = df.set_index("shirt_number")
    assert by_no.loc[1, "position"] == "GK"
    assert by_no.loc[10, "position"] == "FW"
    assert by_no.loc[22, "position"] == "FW"


def test_squad_parser_extracts_birth_date_from_age2_template() -> None:
    df = parse_squad_wikitext(
        _squad_wikitext(),
        team="Argentina",
        tournament="FIFA World Cup 2026",
        snapshot_date=date(2026, 5, 23),
    )
    messi = df[df["player_name"] == "[[Lionel Messi]]"].iloc[0]
    assert messi["birth_date"] == date(1987, 6, 24)


def test_squad_parser_extracts_caps_and_goals_as_ints() -> None:
    df = parse_squad_wikitext(
        _squad_wikitext(),
        team="Argentina",
        tournament="FIFA World Cup 2026",
        snapshot_date=date(2026, 5, 23),
    )
    messi = df[df["player_name"] == "[[Lionel Messi]]"].iloc[0]
    assert messi["caps"] == 190
    assert messi["goals"] == 109


def test_squad_parser_returns_empty_df_when_no_templates() -> None:
    df = parse_squad_wikitext(
        "==Notes==\nNot a squad page.",
        team="X",
        tournament="T",
        snapshot_date=date(2026, 1, 1),
    )
    assert df.empty


def test_fifa_ranking_parser_picks_the_rank_team_points_table() -> None:
    df = parse_fifa_ranking_html(_ranking_html(), ranking_date=date(2026, 4, 4))
    assert list(df["team"]) == ["Argentina", "Spain", "France", "England"]
    assert df["rank"].tolist() == [1, 2, 3, 4]
    assert df["points"].tolist() == [1886.16, 1854.64, 1852.71, 1819.20]
    assert df["previous_rank"].tolist() == [1, 3, 2, 4]


def test_fifa_ranking_parser_returns_empty_on_no_matching_table() -> None:
    df = parse_fifa_ranking_html(
        "<html><body><p>no tables</p></body></html>", ranking_date=date(2026, 4, 4)
    )
    assert df.empty


class _StubResponse:
    def __init__(self, payload: dict[str, Any] | None = None, text: str = ""):
        self._payload = payload or {}
        self.text = text
        self.status_code = 200

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _StubSession:
    """Session lookalike with per-URL stubbed responses."""

    def __init__(self, responses: dict[str, _StubResponse]):
        self._responses = responses
        self.headers: dict[str, str] = {}
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def get(self, url: str, params: dict[str, Any] | None = None, **_):
        self.calls.append((url, params))
        # The MediaWiki API URL is consistent; pick by params.page if present,
        # otherwise the URL itself.
        if params and "page" in params:
            return self._responses.get(params["page"], _StubResponse({}))
        return self._responses.get(url, _StubResponse({}))


def test_fetch_all_squads_writes_combined_parquet(tmp_path: Path) -> None:
    payload = {"parse": {"wikitext": _squad_wikitext()}}
    session = _StubSession({"Argentina at the 2026 FIFA World Cup": _StubResponse(payload=payload)})
    out = fetch_all_squads(
        {"Argentina": "Argentina at the 2026 FIFA World Cup"},
        target_dir=tmp_path,
        session=session,
        today=datetime(2026, 5, 23, tzinfo=UTC),
    )
    assert out.name == "squads_fifa_world_cup_2026_2026-05-23.parquet"
    df = pd.read_parquet(out)
    assert len(df) == 3
    assert set(df["team"]) == {"Argentina"}


def test_fetch_fifa_ranking_writes_parquet(tmp_path: Path) -> None:
    session = _StubSession(
        {
            "https://en.wikipedia.org/wiki/FIFA_Men%27s_World_Ranking": _StubResponse(
                text=_ranking_html()
            )
        }
    )
    out = fetch_fifa_ranking(
        session=session,
        target_dir=tmp_path,
        today=datetime(2026, 4, 4, tzinfo=UTC),
    )
    assert out.name == "fifa_ranking_2026-04-04.parquet"
    df = pd.read_parquet(out)
    assert df["rank"].iloc[0] == 1
    assert df["team"].iloc[0] == "Argentina"


def test_fetch_squad_handles_legacy_wikitext_dict_shape() -> None:
    """MediaWiki formatversion=1 returned wikitext as {"*": ...}; we still support it."""
    from wc2026.ingest.wikipedia import fetch_squad_wikitext

    payload = {"parse": {"wikitext": {"*": _squad_wikitext()}}}
    session = _StubSession({"X": _StubResponse(payload=payload)})
    text = fetch_squad_wikitext("X", session=session)
    assert "Lionel Messi" in text
