"""Live match-state poller — football-data.org → ``raw_live_events``.

The Phase 6 ingester runs every 30 s while a fixture is in its live window
(status ∈ {IN_PLAY, PAUSED, FINISHED}). For each poll we:

1. Pull ``GET /v4/matches/{id}`` from football-data.org.
2. Read the current score + minute + status from the response.
3. Compare against the latest row in ``raw_live_events`` for this ``match_id``.
4. Emit one ``GOAL`` event per score delta and a final ``FT_WHISTLE`` when
   status flips to ``FINISHED``.

Limitations (documented in the Phase 6 README addendum)
-------------------------------------------------------
* football-data.org's free tier does **not** expose detailed events (cards,
  subs, individual goal-scorers), so we only track score deltas.
  ``home_red_cards_after`` / ``away_red_cards_after`` stay at the carried-
  forward values until a higher-fidelity feed is plugged in (TheSportsDB
  timeline, the SportRadar paid feed, etc.).
* The minute we record is the football-data.org ``minute`` value at poll
  time, which lags the live clock by up to one polling interval (30 s).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from wc2026.db.models import RawLiveEvent
from wc2026.db.session import session_scope
from wc2026.ingest.football_data_org import fetch_match

logger = logging.getLogger(__name__)

LIVE_STATUSES: frozenset[str] = frozenset({"IN_PLAY", "PAUSED"})
FINISHED_STATUSES: frozenset[str] = frozenset({"FINISHED"})

EVENT_KICKOFF = "KICKOFF"
EVENT_GOAL = "GOAL"
EVENT_FT_WHISTLE = "FT_WHISTLE"


@dataclass(frozen=True)
class CurrentMatchState:
    """Snapshot of one match's state at poll time."""

    match_id: int
    status: str
    minute: int
    period: int
    home_team: str
    away_team: str
    home_score: int
    away_score: int

    @classmethod
    def from_fdo_payload(cls, payload: dict[str, Any]) -> CurrentMatchState:
        """Adapt the football-data.org ``/matches/{id}`` shape to our schema."""
        score = payload.get("score") or {}
        full_time = score.get("fullTime") or {}
        minute_raw = payload.get("minute")
        if isinstance(minute_raw, str) and minute_raw.isdigit():
            minute = int(minute_raw)
        elif isinstance(minute_raw, int):
            minute = minute_raw
        else:
            minute = 0
        return cls(
            match_id=int(payload.get("id") or 0),
            status=str(payload.get("status") or ""),
            minute=minute,
            period=_period_from_minute(minute),
            home_team=str((payload.get("homeTeam") or {}).get("name") or ""),
            away_team=str((payload.get("awayTeam") or {}).get("name") or ""),
            home_score=int(full_time.get("home") or 0),
            away_score=int(full_time.get("away") or 0),
        )


def _period_from_minute(minute: int) -> int:
    """Best-effort period inference: 1 ≤ minute ≤ 45 → P1, ≤ 90 → P2, then ET."""
    if minute <= 45:
        return 1
    if minute <= 90:
        return 2
    if minute <= 105:
        return 3
    return 4


def _latest_event(session: Session, match_id: int) -> RawLiveEvent | None:
    stmt = (
        select(RawLiveEvent)
        .where(RawLiveEvent.match_id == match_id)
        .order_by(desc(RawLiveEvent.seq))
        .limit(1)
    )
    return session.scalars(stmt).first()


def reconcile_events(
    state: CurrentMatchState,
    *,
    session: Session,
) -> list[RawLiveEvent]:
    """Compute the new ``raw_live_events`` rows implied by a poll.

    Each row is constructed in memory; the caller (``poll_live_match``) is
    responsible for ``session.add_all`` + ``commit``. We split the work so the
    pure-logic part is easy to unit-test against a stub session.
    """
    latest = _latest_event(session, state.match_id)
    prev_home_score = latest.home_score_after if latest else 0
    prev_away_score = latest.away_score_after if latest else 0
    prev_home_reds = latest.home_red_cards_after if latest else 0
    prev_away_reds = latest.away_red_cards_after if latest else 0
    next_seq = (latest.seq + 1) if latest else 1
    now = datetime.now(UTC)

    new_rows: list[RawLiveEvent] = []

    # Seed a KICKOFF row the first time we ever poll a live match.
    if latest is None and (state.status in LIVE_STATUSES or state.status in FINISHED_STATUSES):
        new_rows.append(
            RawLiveEvent(
                match_id=state.match_id,
                seq=next_seq,
                minute=0,
                period=1,
                event_type=EVENT_KICKOFF,
                team=None,
                player=None,
                home_score_after=0,
                away_score_after=0,
                home_red_cards_after=0,
                away_red_cards_after=0,
                ingested_at=now,
            )
        )
        next_seq += 1

    # Emit one GOAL row per score-delta unit on each side. We can't tell from
    # the football-data.org payload alone which side scored when both did
    # between polls — we emit the home goals first, then the away ones.
    for _ in range(max(state.home_score - prev_home_score, 0)):
        prev_home_score += 1
        new_rows.append(
            RawLiveEvent(
                match_id=state.match_id,
                seq=next_seq,
                minute=state.minute,
                period=state.period,
                event_type=EVENT_GOAL,
                team=state.home_team,
                player=None,
                home_score_after=prev_home_score,
                away_score_after=prev_away_score,
                home_red_cards_after=prev_home_reds,
                away_red_cards_after=prev_away_reds,
                ingested_at=now,
            )
        )
        next_seq += 1
    for _ in range(max(state.away_score - prev_away_score, 0)):
        prev_away_score += 1
        new_rows.append(
            RawLiveEvent(
                match_id=state.match_id,
                seq=next_seq,
                minute=state.minute,
                period=state.period,
                event_type=EVENT_GOAL,
                team=state.away_team,
                player=None,
                home_score_after=prev_home_score,
                away_score_after=prev_away_score,
                home_red_cards_after=prev_home_reds,
                away_red_cards_after=prev_away_reds,
                ingested_at=now,
            )
        )
        next_seq += 1

    # Append the FT_WHISTLE once when status flips to FINISHED and the last
    # row we have isn't already that whistle.
    if state.status in FINISHED_STATUSES and (
        latest is None or latest.event_type != EVENT_FT_WHISTLE
    ):
        new_rows.append(
            RawLiveEvent(
                match_id=state.match_id,
                seq=next_seq,
                minute=state.minute or 90,
                period=state.period,
                event_type=EVENT_FT_WHISTLE,
                team=None,
                player=None,
                home_score_after=state.home_score,
                away_score_after=state.away_score,
                home_red_cards_after=prev_home_reds,
                away_red_cards_after=prev_away_reds,
                ingested_at=now,
            )
        )

    return new_rows


def poll_live_match(
    match_id: int,
    *,
    engine: Engine | None = None,
    session_factory=None,
    api_key: str | None = None,
    fetch_func=fetch_match,
) -> int:
    """Polite one-shot poll: fetch state, reconcile, persist. Returns # new rows.

    No-ops (returns 0) when the match status isn't IN_PLAY / PAUSED / FINISHED
    — we don't want to seed kickoff rows for matches that haven't started.
    """
    try:
        payload = fetch_func(match_id, api_key=api_key)
    except Exception:
        logger.exception("poll_live_match(%d): football-data.org fetch failed", match_id)
        return 0
    state = CurrentMatchState.from_fdo_payload(payload)
    if state.status not in LIVE_STATUSES and state.status not in FINISHED_STATUSES:
        return 0

    def _do_with(session: Session) -> int:
        new_rows = reconcile_events(state, session=session)
        if not new_rows:
            return 0
        session.add_all(new_rows)
        return len(new_rows)

    if session_factory is not None:
        with session_factory() as session:  # caller-supplied
            n = _do_with(session)
            session.commit()
            return n
    db_url = engine.url.render_as_string(hide_password=False) if engine is not None else None
    with session_scope(db_url) as session:
        return _do_with(session)


def load_event_history(
    match_id: int,
    *,
    engine: Engine | None = None,
) -> list[RawLiveEvent]:
    """Return every persisted event for ``match_id``, ordered by ``seq``."""
    db_url = engine.url.render_as_string(hide_password=False) if engine is not None else None
    with session_scope(db_url) as session:
        stmt = (
            select(RawLiveEvent).where(RawLiveEvent.match_id == match_id).order_by(RawLiveEvent.seq)
        )
        rows = list(session.scalars(stmt))
        # Detach from the session so callers can use the objects after the scope closes.
        for r in rows:
            session.expunge(r)
        return rows


__all__ = [
    "EVENT_FT_WHISTLE",
    "EVENT_GOAL",
    "EVENT_KICKOFF",
    "FINISHED_STATUSES",
    "LIVE_STATUSES",
    "CurrentMatchState",
    "load_event_history",
    "poll_live_match",
    "reconcile_events",
]
