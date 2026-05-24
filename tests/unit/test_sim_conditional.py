"""Unit tests for the conditional-MC helper that bridges live events → sim."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from wc2026.db.models import Base, RawLiveEvent
from wc2026.sim.conditional import known_group_results_from_live_events


@pytest.fixture
def sqlite_engine():
    eng = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(eng)
    try:
        yield eng
    finally:
        eng.dispose()


def _ft_whistle(*, match_id: int, home_score: int, away_score: int):
    return RawLiveEvent(
        match_id=match_id,
        seq=99,
        minute=90,
        period=2,
        event_type="FT_WHISTLE",
        team=None,
        player=None,
        home_score_after=home_score,
        away_score_after=away_score,
        home_red_cards_after=0,
        away_red_cards_after=0,
        ingested_at=datetime.now(UTC),
    )


def test_returns_empty_when_mapping_is_empty(sqlite_engine) -> None:
    assert known_group_results_from_live_events({}, engine=sqlite_engine) == {}


def test_returns_empty_when_no_ft_whistle_rows(sqlite_engine) -> None:
    mapping = {1: (date(2026, 6, 11), "Mexico", "Senegal")}
    assert known_group_results_from_live_events(mapping, engine=sqlite_engine) == {}


def test_translates_match_id_via_mapping(sqlite_engine) -> None:
    with Session(sqlite_engine, future=True) as session:
        session.add(_ft_whistle(match_id=42, home_score=2, away_score=1))
        session.commit()
    mapping = {42: (date(2026, 6, 11), "Mexico", "Senegal")}
    out = known_group_results_from_live_events(mapping, engine=sqlite_engine)
    assert out == {("Mexico", "Senegal"): (2, 1)}


def test_silently_skips_match_ids_not_in_mapping(sqlite_engine) -> None:
    """Defense against a stale football-data.org cache."""
    with Session(sqlite_engine, future=True) as session:
        session.add(_ft_whistle(match_id=99, home_score=3, away_score=0))
        session.commit()
    mapping = {42: (date(2026, 6, 11), "Mexico", "Senegal")}
    assert known_group_results_from_live_events(mapping, engine=sqlite_engine) == {}


def test_ignores_non_ft_whistle_rows(sqlite_engine) -> None:
    """KICKOFF / GOAL rows must NOT show up in the known-results map."""
    with Session(sqlite_engine, future=True) as session:
        session.add(
            RawLiveEvent(
                match_id=42,
                seq=1,
                minute=0,
                period=1,
                event_type="KICKOFF",
                team=None,
                player=None,
                home_score_after=0,
                away_score_after=0,
                home_red_cards_after=0,
                away_red_cards_after=0,
                ingested_at=datetime.now(UTC),
            )
        )
        session.add(
            RawLiveEvent(
                match_id=42,
                seq=2,
                minute=23,
                period=1,
                event_type="GOAL",
                team="Mexico",
                player=None,
                home_score_after=1,
                away_score_after=0,
                home_red_cards_after=0,
                away_red_cards_after=0,
                ingested_at=datetime.now(UTC),
            )
        )
        session.commit()
    mapping = {42: (date(2026, 6, 11), "Mexico", "Senegal")}
    # No FT_WHISTLE row → no known result.
    assert known_group_results_from_live_events(mapping, engine=sqlite_engine) == {}


def test_handles_multiple_completed_matches(sqlite_engine) -> None:
    with Session(sqlite_engine, future=True) as session:
        session.add(_ft_whistle(match_id=1, home_score=2, away_score=1))
        session.add(_ft_whistle(match_id=2, home_score=0, away_score=0))
        session.add(_ft_whistle(match_id=3, home_score=1, away_score=3))
        session.commit()
    mapping = {
        1: (date(2026, 6, 11), "Mexico", "Senegal"),
        2: (date(2026, 6, 12), "Argentina", "France"),
        3: (date(2026, 6, 13), "Brazil", "Germany"),
    }
    out = known_group_results_from_live_events(mapping, engine=sqlite_engine)
    assert out == {
        ("Mexico", "Senegal"): (2, 1),
        ("Argentina", "France"): (0, 0),
        ("Brazil", "Germany"): (1, 3),
    }
