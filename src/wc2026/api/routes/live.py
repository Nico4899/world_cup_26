"""Live in-match endpoints: snapshot + Server-Sent Events stream.

``GET /api/v1/live/{match_id}``       — JSON snapshot used by Streamlit polling.
``GET /api/v1/live/{match_id}/sse``   — SSE stream for any client that prefers
                                         push over poll.

Both endpoints degrade gracefully when the Phase 6 live model isn't loaded:
they return the pre-match Poisson probability and a ``win_prob_source`` of
``"poisson_pre_match"`` so callers can flag the fallback.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from wc2026.api.dependencies import get_fixtures, get_model
from wc2026.api.schemas import (
    LiveEventTrace,
    LiveHistory,
    LiveStateSnapshot,
    OutcomeProbabilities,
)
from wc2026.db.models import RawLiveEvent
from wc2026.db.session import session_scope
from wc2026.ingest.live_events import EVENT_FT_WHISTLE, EVENT_KICKOFF
from wc2026.models.live_win_prob import (
    LiveWinProbModel,
    minutes_remaining_from_minute,
)
from wc2026.models.poisson_dc import PoissonDC
from wc2026.sim.fixtures import FixtureMatch, WC2026Fixtures

router = APIRouter(prefix="/api/v1/live")
logger = logging.getLogger(__name__)

DEFAULT_SSE_POLL_INTERVAL_SECONDS = 5.0


def _resolve_fixture(fixtures: WC2026Fixtures, match_id: int) -> FixtureMatch:
    if match_id < 0 or match_id >= len(fixtures.matches):
        raise HTTPException(
            status_code=404,
            detail=f"match_id {match_id} out of range (0..{len(fixtures.matches) - 1})",
        )
    return fixtures.matches[match_id]


def _latest_event(match_id: int) -> RawLiveEvent | None:
    try:
        with session_scope() as session:
            row = session.scalars(
                select(RawLiveEvent)
                .where(RawLiveEvent.match_id == match_id)
                .order_by(RawLiveEvent.seq.desc())
                .limit(1)
            ).first()
            if row is not None:
                session.expunge(row)
            return row
    except Exception:
        # DB unreachable (no Postgres in dev). Treat as "no live state".
        logger.debug("live: DB unreachable for /live history", exc_info=True)
        return None


def _all_events(match_id: int) -> list[RawLiveEvent]:
    try:
        with session_scope() as session:
            rows = list(
                session.scalars(
                    select(RawLiveEvent)
                    .where(RawLiveEvent.match_id == match_id)
                    .order_by(RawLiveEvent.seq)
                )
            )
            for r in rows:
                session.expunge(r)
            return rows
    except Exception:
        logger.debug("live: DB unreachable for /live SSE", exc_info=True)
        return []


def _elo_diff_for(request: Request, home_team: str, away_team: str) -> float:
    sources = getattr(request.app.state, "feature_sources", None)
    if sources is None or not sources.elo_by_team:
        return 0.0
    h = sources.elo_by_team.get(home_team)
    a = sources.elo_by_team.get(away_team)
    if h is None or a is None:
        return 0.0
    return float(h) - float(a)


def _win_prob_for_state(
    *,
    live_model: LiveWinProbModel | None,
    poisson_model: PoissonDC,
    fixture: FixtureMatch,
    event: RawLiveEvent | None,
    elo_diff: float,
) -> tuple[OutcomeProbabilities, str]:
    """Pick the right model + return ``(win_prob, source_tag)``."""
    if event is not None and event.event_type == EVENT_FT_WHISTLE:
        if event.home_score_after > event.away_score_after:
            return OutcomeProbabilities(home_win=1.0, draw=0.0, away_win=0.0), "final"
        if event.home_score_after < event.away_score_after:
            return OutcomeProbabilities(home_win=0.0, draw=0.0, away_win=1.0), "final"
        return OutcomeProbabilities(home_win=0.0, draw=1.0, away_win=0.0), "final"
    if live_model is None or event is None or event.event_type == EVENT_KICKOFF:
        # No model yet, or no in-running state: use the Poisson pre-match.
        poisson_outcome = poisson_model.outcome_probs(
            fixture.home_team, fixture.away_team, neutral=fixture.neutral
        )
        return (
            OutcomeProbabilities(
                home_win=float(poisson_outcome["home_win"]),
                draw=float(poisson_outcome["draw"]),
                away_win=float(poisson_outcome["away_win"]),
            ),
            "poisson_pre_match",
        )
    goal_diff = event.home_score_after - event.away_score_after
    red_diff = event.home_red_cards_after - event.away_red_cards_after
    minutes_remaining = minutes_remaining_from_minute(event.minute, event.period)
    triplet = live_model.predict_one(
        elo_diff=elo_diff,
        goal_diff=goal_diff,
        minutes_remaining=minutes_remaining,
        red_diff=red_diff,
    )
    return OutcomeProbabilities(**triplet), "live_win_prob"


def _snapshot_for_event(
    *,
    match_id: int,
    fixture: FixtureMatch,
    event: RawLiveEvent | None,
    win_prob: OutcomeProbabilities,
    source: str,
) -> LiveStateSnapshot:
    if event is None:
        return LiveStateSnapshot(
            match_id=match_id,
            home_team=fixture.home_team,
            away_team=fixture.away_team,
            minute=0,
            period=1,
            home_score=0,
            away_score=0,
            home_red_cards=0,
            away_red_cards=0,
            last_event_type=EVENT_KICKOFF,
            win_prob=win_prob,
            win_prob_source=source,
        )
    return LiveStateSnapshot(
        match_id=match_id,
        home_team=fixture.home_team,
        away_team=fixture.away_team,
        minute=event.minute,
        period=event.period,
        home_score=event.home_score_after,
        away_score=event.away_score_after,
        home_red_cards=event.home_red_cards_after,
        away_red_cards=event.away_red_cards_after,
        last_event_type=event.event_type,
        win_prob=win_prob,
        win_prob_source=source,
    )


@router.get("/{match_id}", response_model=LiveStateSnapshot)
def live_snapshot(
    match_id: int,
    request: Request,
    poisson_model: PoissonDC = Depends(get_model),
    fixtures: WC2026Fixtures = Depends(get_fixtures),
) -> LiveStateSnapshot:
    """Return the current state + win-prob for one WC 2026 fixture."""
    fixture = _resolve_fixture(fixtures, match_id)
    event = _latest_event(match_id)
    live_model = getattr(request.app.state, "live_win_prob_model", None)
    elo_diff = _elo_diff_for(request, fixture.home_team, fixture.away_team)
    win_prob, source = _win_prob_for_state(
        live_model=live_model,
        poisson_model=poisson_model,
        fixture=fixture,
        event=event,
        elo_diff=elo_diff,
    )
    return _snapshot_for_event(
        match_id=match_id,
        fixture=fixture,
        event=event,
        win_prob=win_prob,
        source=source,
    )


@router.get("/{match_id}/history", response_model=LiveHistory)
def live_history(
    match_id: int,
    request: Request,
    poisson_model: PoissonDC = Depends(get_model),
    fixtures: WC2026Fixtures = Depends(get_fixtures),
) -> LiveHistory:
    """Return every observed event + the current snapshot. Used by the dashboard chart."""
    fixture = _resolve_fixture(fixtures, match_id)
    events = _all_events(match_id)
    live_model = getattr(request.app.state, "live_win_prob_model", None)
    elo_diff = _elo_diff_for(request, fixture.home_team, fixture.away_team)
    traces: list[LiveEventTrace] = []
    for ev in events:
        win_prob, _ = _win_prob_for_state(
            live_model=live_model,
            poisson_model=poisson_model,
            fixture=fixture,
            event=ev,
            elo_diff=elo_diff,
        )
        traces.append(
            LiveEventTrace(
                seq=ev.seq,
                minute=ev.minute,
                period=ev.period,
                event_type=ev.event_type,
                team=ev.team,
                home_score_after=ev.home_score_after,
                away_score_after=ev.away_score_after,
                home_red_cards_after=ev.home_red_cards_after,
                away_red_cards_after=ev.away_red_cards_after,
                win_prob=win_prob,
            )
        )
    latest_event = events[-1] if events else None
    snapshot_win_prob, source = _win_prob_for_state(
        live_model=live_model,
        poisson_model=poisson_model,
        fixture=fixture,
        event=latest_event,
        elo_diff=elo_diff,
    )
    snapshot = _snapshot_for_event(
        match_id=match_id,
        fixture=fixture,
        event=latest_event,
        win_prob=snapshot_win_prob,
        source=source,
    )
    return LiveHistory(snapshot=snapshot, events=traces)


def _format_sse(payload: dict[str, Any]) -> bytes:
    """Format an SSE ``data: ...`` chunk; returns bytes ready for the wire."""
    return f"data: {json.dumps(payload, default=str)}\n\n".encode()


async def _stream_match_events(
    *,
    request: Request,
    match_id: int,
    fixture: FixtureMatch,
    elo_diff: float,
    poisson_model: PoissonDC,
    live_model: LiveWinProbModel | None,
    poll_interval: float,
) -> AsyncGenerator[bytes, None]:
    """Async generator: emit one SSE chunk per known event, then poll for new ones.

    Stops when the client disconnects or when the most-recent emitted event is
    ``FT_WHISTLE``.
    """
    emitted: set[int] = set()

    def emit_chunks(events: list[RawLiveEvent]) -> list[bytes]:
        chunks: list[bytes] = []
        for ev in events:
            if ev.seq in emitted:
                continue
            win_prob, source = _win_prob_for_state(
                live_model=live_model,
                poisson_model=poisson_model,
                fixture=fixture,
                event=ev,
                elo_diff=elo_diff,
            )
            snapshot = _snapshot_for_event(
                match_id=match_id,
                fixture=fixture,
                event=ev,
                win_prob=win_prob,
                source=source,
            )
            chunks.append(_format_sse(snapshot.model_dump(mode="json")))
            emitted.add(ev.seq)
        return chunks

    # Initial flush: every event already on disk.
    history = _all_events(match_id)
    if not history:
        # No state yet — emit one pre-match "kickoff-equivalent" snapshot so the
        # client always sees at least one frame.
        win_prob, source = _win_prob_for_state(
            live_model=live_model,
            poisson_model=poisson_model,
            fixture=fixture,
            event=None,
            elo_diff=elo_diff,
        )
        snapshot = _snapshot_for_event(
            match_id=match_id,
            fixture=fixture,
            event=None,
            win_prob=win_prob,
            source=source,
        )
        yield _format_sse(snapshot.model_dump(mode="json"))
    else:
        for chunk in emit_chunks(history):
            yield chunk
        if history[-1].event_type == EVENT_FT_WHISTLE:
            return

    # Poll loop.
    while True:
        try:
            disconnected = await request.is_disconnected()
        except Exception:
            disconnected = False
        if disconnected:
            return
        try:
            await asyncio.sleep(poll_interval)
        except asyncio.CancelledError:
            return
        new_events = _all_events(match_id)
        for chunk in emit_chunks(new_events):
            yield chunk
        if new_events and new_events[-1].event_type == EVENT_FT_WHISTLE:
            return


@router.get("/{match_id}/sse")
async def live_sse(
    match_id: int,
    request: Request,
    poisson_model: PoissonDC = Depends(get_model),
    fixtures: WC2026Fixtures = Depends(get_fixtures),
    poll_interval: float = DEFAULT_SSE_POLL_INTERVAL_SECONDS,
) -> StreamingResponse:
    """Server-Sent Events stream of live win-prob snapshots for one fixture."""
    fixture = _resolve_fixture(fixtures, match_id)
    live_model = getattr(request.app.state, "live_win_prob_model", None)
    elo_diff = _elo_diff_for(request, fixture.home_team, fixture.away_team)
    return StreamingResponse(
        _stream_match_events(
            request=request,
            match_id=match_id,
            fixture=fixture,
            elo_diff=elo_diff,
            poisson_model=poisson_model,
            live_model=live_model,
            poll_interval=poll_interval,
        ),
        media_type="text/event-stream",
    )


__all__ = ["router"]
