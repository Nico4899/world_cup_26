"""Unit tests for the live-event poller + reconciler.

The poller's pure-logic ``reconcile_events`` is tested against an in-memory
SQLite engine so we don't need Postgres; the HTTP fetcher is stubbed end-to-
end via the ``fetch_func`` injection point on ``poll_live_match``.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from wc2026.db.models import Base, RawLiveEvent
from wc2026.ingest.live_events import (
    EVENT_FT_WHISTLE,
    EVENT_GOAL,
    EVENT_KICKOFF,
    CurrentMatchState,
    poll_live_match,
    reconcile_events,
)


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(eng)
    try:
        yield eng
    finally:
        eng.dispose()


def _state(**overrides) -> CurrentMatchState:
    defaults = {
        "match_id": 1,
        "status": "IN_PLAY",
        "minute": 23,
        "period": 1,
        "home_team": "Argentina",
        "away_team": "France",
        "home_score": 0,
        "away_score": 0,
    }
    defaults.update(overrides)
    return CurrentMatchState(**defaults)


def test_from_fdo_payload_extracts_canonical_fields() -> None:
    payload = {
        "id": 7,
        "status": "IN_PLAY",
        "minute": "62",
        "homeTeam": {"name": "Argentina"},
        "awayTeam": {"name": "France"},
        "score": {"fullTime": {"home": 1, "away": 1}},
    }
    state = CurrentMatchState.from_fdo_payload(payload)
    assert state.match_id == 7
    assert state.minute == 62
    assert state.period == 2  # minute > 45
    assert state.home_score == 1
    assert state.away_score == 1


def test_reconcile_seeds_kickoff_on_first_poll_in_play(engine) -> None:
    with Session(engine) as session:
        rows = reconcile_events(_state(home_score=0, away_score=0), session=session)
        session.add_all(rows)
        session.commit()
        loaded = list(session.scalars(select(RawLiveEvent).order_by(RawLiveEvent.seq)))
        assert len(loaded) == 1
        assert loaded[0].event_type == EVENT_KICKOFF


def test_reconcile_no_ops_on_subsequent_poll_with_no_change(engine) -> None:
    state = _state(home_score=0, away_score=0)
    with Session(engine) as session:
        session.add_all(reconcile_events(state, session=session))
        session.commit()
        rows = reconcile_events(state, session=session)
        assert rows == []


def test_reconcile_emits_one_goal_per_home_delta(engine) -> None:
    with Session(engine) as session:
        # First poll: seed kickoff.
        session.add_all(reconcile_events(_state(home_score=0, away_score=0), session=session))
        session.commit()
        # Second poll: 1-0 home — one GOAL row for Argentina.
        rows = reconcile_events(
            _state(home_score=1, away_score=0, minute=23), session=session
        )
        assert len(rows) == 1
        assert rows[0].event_type == EVENT_GOAL
        assert rows[0].team == "Argentina"
        assert rows[0].home_score_after == 1
        assert rows[0].away_score_after == 0


def test_reconcile_emits_multiple_goals_when_polls_lag(engine) -> None:
    with Session(engine) as session:
        session.add_all(reconcile_events(_state(home_score=0, away_score=0), session=session))
        session.commit()
        # Two goals between polls: 0-0 → 2-1.
        rows = reconcile_events(
            _state(home_score=2, away_score=1, minute=37), session=session
        )
        # Two home goals then one away goal.
        assert [r.event_type for r in rows] == [EVENT_GOAL, EVENT_GOAL, EVENT_GOAL]
        assert [r.team for r in rows] == ["Argentina", "Argentina", "France"]
        # Sequence numbers are monotonic.
        assert [r.seq for r in rows] == [2, 3, 4]
        # Score progressively populates the after-state.
        assert rows[0].home_score_after == 1
        assert rows[1].home_score_after == 2
        assert rows[2].away_score_after == 1


def test_reconcile_emits_ft_whistle_when_status_finished(engine) -> None:
    with Session(engine) as session:
        session.add_all(reconcile_events(_state(home_score=2, away_score=1), session=session))
        session.commit()
        rows = reconcile_events(
            _state(
                home_score=2, away_score=1, minute=90, period=2, status="FINISHED"
            ),
            session=session,
        )
        # No new goals; one FT_WHISTLE.
        assert len(rows) == 1
        assert rows[0].event_type == EVENT_FT_WHISTLE


def test_reconcile_does_not_double_emit_ft_whistle(engine) -> None:
    with Session(engine) as session:
        session.add_all(reconcile_events(_state(home_score=2, away_score=1), session=session))
        session.commit()
        first = reconcile_events(
            _state(
                home_score=2, away_score=1, minute=90, period=2, status="FINISHED"
            ),
            session=session,
        )
        session.add_all(first)
        session.commit()
        # Second FINISHED poll: should be empty (no re-emission).
        second = reconcile_events(
            _state(
                home_score=2, away_score=1, minute=90, period=2, status="FINISHED"
            ),
            session=session,
        )
        assert second == []


def test_poll_live_match_no_ops_when_status_scheduled(engine) -> None:
    """A SCHEDULED match shouldn't even seed a kickoff."""
    sm = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    payload = {
        "id": 1,
        "status": "SCHEDULED",
        "homeTeam": {"name": "Argentina"},
        "awayTeam": {"name": "France"},
        "score": {"fullTime": {"home": None, "away": None}},
    }
    n = poll_live_match(1, session_factory=sm, fetch_func=lambda *a, **k: payload)
    assert n == 0
    with Session(engine) as session:
        assert list(session.scalars(select(RawLiveEvent))) == []


def test_poll_live_match_persists_kickoff_then_goal(engine) -> None:
    """Two sequential polls: kickoff seed, then 1-0 → emits a goal row."""
    sm = sessionmaker(bind=engine, future=True, expire_on_commit=False)

    p_zero = {
        "id": 1,
        "status": "IN_PLAY",
        "minute": "5",
        "homeTeam": {"name": "Argentina"},
        "awayTeam": {"name": "France"},
        "score": {"fullTime": {"home": 0, "away": 0}},
    }
    assert poll_live_match(1, session_factory=sm, fetch_func=lambda *a, **k: p_zero) == 1

    p_one = {**p_zero, "minute": "23", "score": {"fullTime": {"home": 1, "away": 0}}}
    assert poll_live_match(1, session_factory=sm, fetch_func=lambda *a, **k: p_one) == 1

    with Session(engine) as session:
        rows = list(session.scalars(select(RawLiveEvent).order_by(RawLiveEvent.seq)))
        assert [r.event_type for r in rows] == [EVENT_KICKOFF, EVENT_GOAL]


def test_poll_live_match_returns_zero_on_http_failure(engine) -> None:
    sm = sessionmaker(bind=engine, future=True, expire_on_commit=False)

    def boom(*a, **k):
        raise RuntimeError("boom")

    assert poll_live_match(1, session_factory=sm, fetch_func=boom) == 0
