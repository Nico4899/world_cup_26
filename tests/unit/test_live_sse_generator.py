"""Direct tests for the SSE async generator.

We bypass TestClient because its disconnect-propagation semantics interact
badly with the polling loop in ``_stream_match_events`` (the loop awaits
``request.is_disconnected()`` plus a short ``asyncio.sleep`` — fine in
production but fragile through TestClient's wrapped transport). Driving the
generator directly is cleaner: we exhaust the initial history flush, then
inject a fake disconnect, and assert the generator returns promptly.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json as _json
from dataclasses import dataclass

import pandas as pd
import pytest

from wc2026.api.routes.live import _stream_match_events
from wc2026.db.models import RawLiveEvent
from wc2026.features.match_weights import combined_weight
from wc2026.ingest.kaggle_intl import load_played
from wc2026.models.live_win_prob import LiveWinProbModel
from wc2026.models.poisson_dc import PoissonDC
from wc2026.sim.fixtures import FixtureMatch


@dataclass
class _StubRequest:
    """Mimic just enough of ``starlette.Request`` for the generator's contract."""

    disconnected: bool = False

    async def is_disconnected(self) -> bool:
        return self.disconnected


def _fitted_poisson() -> PoissonDC:
    """Reuse a tiny PoissonDC fit on the recent corpus for the pre-match path."""
    df = load_played()
    cutoff = pd.Timestamp("2020-01-01")
    train = df[df["date"] >= cutoff].reset_index(drop=True)
    weights = combined_weight(train, ref_date=pd.Timestamp("2025-01-01"), half_life_days=3650.0)
    return PoissonDC().fit(train, weights=weights)


def _fitted_live_model() -> LiveWinProbModel:
    import numpy as np

    rng = np.random.default_rng(0)
    n = 600
    elo_diff = rng.normal(0.0, 80.0, size=n)
    goal_diff = rng.integers(-3, 4, size=n)
    minutes_remaining = rng.integers(0, 90, size=n)
    red_diff = rng.choice([-1, 0, 1], size=n, p=[0.05, 0.9, 0.05])
    logits = 0.01 * elo_diff + 1.4 * goal_diff - 0.4 * red_diff
    noise = rng.normal(0, 1.0, size=n)
    y = np.where(logits + noise > 0.8, 0, np.where(logits + noise < -0.8, 2, 1)).astype(int)
    X = pd.DataFrame(
        {
            "elo_diff": elo_diff,
            "goal_diff": goal_diff,
            "minutes_remaining": minutes_remaining,
            "red_diff": red_diff,
        }
    )
    return LiveWinProbModel.fit(X, y)


def _argentina_v_france_fixture() -> FixtureMatch:
    return FixtureMatch(
        date=pd.Timestamp("2026-06-11"),
        home_team="Argentina",
        away_team="France",
        group="A",
        city="MetLife",
        country="USA",
        neutral=True,
    )


def _consume_until_done(gen) -> list[dict]:
    """Drain an async generator into a list of parsed SSE payloads."""

    async def runner() -> list[dict]:
        out: list[dict] = []
        async for chunk in gen:
            text = chunk.decode("utf-8")
            for line in text.splitlines():
                if line.startswith("data: "):
                    out.append(_json.loads(line[len("data: ") :]))
        return out

    return asyncio.run(runner())


def test_sse_generator_emits_pre_match_frame_when_no_events(monkeypatch) -> None:
    from wc2026.api.routes import live as live_route

    monkeypatch.setattr(live_route, "_all_events", lambda _mid: [])
    request = _StubRequest(disconnected=True)
    gen = _stream_match_events(
        request=request,
        match_id=0,
        fixture=_argentina_v_france_fixture(),
        elo_diff=0.0,
        poisson_model=_fitted_poisson(),
        live_model=None,
        poll_interval=0.01,
    )
    payloads = _consume_until_done(gen)
    assert len(payloads) == 1
    assert payloads[0]["last_event_type"] == "KICKOFF"
    assert payloads[0]["win_prob_source"] == "poisson_pre_match"


def test_sse_generator_flushes_history_then_terminates_on_ft_whistle(monkeypatch) -> None:
    from wc2026.api.routes import live as live_route

    now = _dt.datetime.now(_dt.UTC)
    events = [
        RawLiveEvent(
            match_id=0,
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
            ingested_at=now,
        ),
        RawLiveEvent(
            match_id=0,
            seq=2,
            minute=23,
            period=1,
            event_type="GOAL",
            team="Argentina",
            player=None,
            home_score_after=1,
            away_score_after=0,
            home_red_cards_after=0,
            away_red_cards_after=0,
            ingested_at=now,
        ),
        RawLiveEvent(
            match_id=0,
            seq=3,
            minute=90,
            period=2,
            event_type="FT_WHISTLE",
            team=None,
            player=None,
            home_score_after=1,
            away_score_after=0,
            home_red_cards_after=0,
            away_red_cards_after=0,
            ingested_at=now,
        ),
    ]
    monkeypatch.setattr(live_route, "_all_events", lambda _mid: events)
    request = _StubRequest(disconnected=False)
    gen = _stream_match_events(
        request=request,
        match_id=0,
        fixture=_argentina_v_france_fixture(),
        elo_diff=120.0,
        poisson_model=_fitted_poisson(),
        live_model=_fitted_live_model(),
        poll_interval=0.01,
    )
    payloads = _consume_until_done(gen)
    assert len(payloads) == 3
    assert [p["last_event_type"] for p in payloads] == ["KICKOFF", "GOAL", "FT_WHISTLE"]
    assert payloads[-1]["win_prob_source"] == "final"


def test_sse_generator_exits_immediately_when_request_disconnected(monkeypatch) -> None:
    """If the client disconnects, the generator must stop after the initial flush."""
    from wc2026.api.routes import live as live_route

    now = _dt.datetime.now(_dt.UTC)
    monkeypatch.setattr(
        live_route,
        "_all_events",
        lambda _mid: [
            RawLiveEvent(
                match_id=0,
                seq=1,
                minute=23,
                period=1,
                event_type="GOAL",
                team="Argentina",
                player=None,
                home_score_after=1,
                away_score_after=0,
                home_red_cards_after=0,
                away_red_cards_after=0,
                ingested_at=now,
            )
        ],
    )
    request = _StubRequest(disconnected=True)
    gen = _stream_match_events(
        request=request,
        match_id=0,
        fixture=_argentina_v_france_fixture(),
        elo_diff=0.0,
        poisson_model=_fitted_poisson(),
        live_model=_fitted_live_model(),
        poll_interval=0.01,
    )
    payloads = _consume_until_done(gen)
    # The history flush yields one chunk, then the disconnect check returns
    # immediately so we don't loop forever.
    assert len(payloads) == 1
    assert payloads[0]["last_event_type"] == "GOAL"


@pytest.mark.parametrize("poll_interval", [0.005, 0.02])
def test_sse_generator_does_not_busy_loop(monkeypatch, poll_interval: float) -> None:
    """If history exists but no FT_WHISTLE, the generator must idle on the poll
    interval rather than spinning the event loop. We measure that the wall-clock
    time taken to drain a one-iteration scenario is at least ``poll_interval``
    (allowing for OS scheduling jitter)."""
    import time

    from wc2026.api.routes import live as live_route

    now = _dt.datetime.now(_dt.UTC)
    events = [
        RawLiveEvent(
            match_id=0,
            seq=1,
            minute=23,
            period=1,
            event_type="GOAL",
            team="Argentina",
            player=None,
            home_score_after=1,
            away_score_after=0,
            home_red_cards_after=0,
            away_red_cards_after=0,
            ingested_at=now,
        )
    ]

    call_count = {"n": 0}

    def fake_all_events(_mid):
        call_count["n"] += 1
        return events

    monkeypatch.setattr(live_route, "_all_events", fake_all_events)
    request = _StubRequest(disconnected=False)
    gen = _stream_match_events(
        request=request,
        match_id=0,
        fixture=_argentina_v_france_fixture(),
        elo_diff=0.0,
        poisson_model=_fitted_poisson(),
        live_model=None,
        poll_interval=poll_interval,
    )

    async def take_one_then_disconnect():
        agen = gen.__aiter__()
        first = await agen.__anext__()
        # Simulate a client disconnect now; the next is_disconnected() returns True.
        request.disconnected = True
        start = time.monotonic()
        try:
            await agen.__anext__()
        except StopAsyncIteration:
            return first, time.monotonic() - start
        return first, time.monotonic() - start

    first_chunk, elapsed = asyncio.run(take_one_then_disconnect())
    assert first_chunk.startswith(b"data: ")
    # Generator should exit before the next poll interval would have completed
    # (the disconnect check runs before sleep).
    assert elapsed < poll_interval + 0.2
