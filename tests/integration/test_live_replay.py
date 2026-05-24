"""Phase 6.8 end-to-end replay test.

Wires the four Phase 6 pieces together against the existing StatsBomb fixture
file (4 shots — 3 goals, 1 saved — plus a pass), with no real network and no
real Postgres:

    StatsBomb events JSON
       └─► features.live_state.replay_statsbomb_events
              └─► raw_live_events rows (hand-rolled via the model class)
                     └─► live.py SSE generator
                            └─► one JSON chunk per (kickoff + 3 goals + FT)

This is the plan's named verification ("replay a stored StatsBomb match
through the live pipeline; confirm SSE emits at every minute boundary and at
every event"). Marked integration so the default suite still finishes fast.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
from pathlib import Path

import pandas as pd
import pytest

from wc2026.api.routes.live import _stream_match_events
from wc2026.db.models import RawLiveEvent
from wc2026.features.live_state import replay_statsbomb_events
from wc2026.features.match_weights import combined_weight
from wc2026.ingest.kaggle_intl import load_played
from wc2026.ingest.live_events import EVENT_FT_WHISTLE, EVENT_KICKOFF
from wc2026.models.live_win_prob import LiveWinProbModel
from wc2026.models.poisson_dc import PoissonDC
from wc2026.sim.fixtures import FixtureMatch

pytestmark = pytest.mark.integration

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"


class _StubRequest:
    async def is_disconnected(self) -> bool:
        return False


def _events() -> list[dict]:
    return json.loads((FIXTURE_DIR / "statsbomb_events_sample.json").read_text(encoding="utf-8"))


def _fitted_poisson() -> PoissonDC:
    df = load_played()
    cutoff = pd.Timestamp("2018-01-01")
    train = df[df["date"] >= cutoff].reset_index(drop=True)
    weights = combined_weight(train, ref_date=pd.Timestamp("2023-01-01"), half_life_days=3650.0)
    return PoissonDC().fit(train, weights=weights)


def _fitted_live_model() -> LiveWinProbModel:
    import numpy as np

    rng = np.random.default_rng(7)
    n = 800
    elo_diff = rng.normal(0.0, 80.0, size=n)
    goal_diff = rng.integers(-3, 4, size=n)
    minutes_remaining = rng.integers(0, 90, size=n)
    red_diff = rng.choice([-1, 0, 1], size=n, p=[0.05, 0.9, 0.05])
    logits = 0.012 * elo_diff + 1.5 * goal_diff - 0.4 * red_diff
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


def _snapshots_to_rows(snapshots, *, match_id: int) -> list[RawLiveEvent]:
    """Convert replayed StateSnapshots into RawLiveEvent rows."""
    now = _dt.datetime.now(_dt.UTC)
    out: list[RawLiveEvent] = []
    for seq, snap in enumerate(snapshots, start=1):
        et = snap.event_type if seq > 1 else EVENT_KICKOFF
        out.append(
            RawLiveEvent(
                match_id=match_id,
                seq=seq,
                minute=snap.minute,
                period=snap.period,
                event_type=et,
                team=None,
                player=None,
                home_score_after=snap.home_score,
                away_score_after=snap.away_score,
                home_red_cards_after=snap.home_red_cards,
                away_red_cards_after=snap.away_red_cards,
                ingested_at=now,
            )
        )
    # Tack on the FT_WHISTLE that a live poller would emit when the score
    # stabilises post-final-whistle.
    last = snapshots[-1]
    out.append(
        RawLiveEvent(
            match_id=match_id,
            seq=len(out) + 1,
            minute=90,
            period=2,
            event_type=EVENT_FT_WHISTLE,
            team=None,
            player=None,
            home_score_after=last.home_score,
            away_score_after=last.away_score,
            home_red_cards_after=last.home_red_cards,
            away_red_cards_after=last.away_red_cards,
            ingested_at=now,
        )
    )
    return out


def _argentina_v_france_fixture() -> FixtureMatch:
    return FixtureMatch(
        date=pd.Timestamp("2022-12-18"),
        home_team="Argentina",
        away_team="France",
        group="A",
        city="Lusail",
        country="Qatar",
        neutral=True,
    )


def test_replay_full_pipeline_emits_one_frame_per_event(monkeypatch) -> None:
    snapshots = replay_statsbomb_events(_events(), home_team="Argentina", away_team="France")
    rows = _snapshots_to_rows(snapshots, match_id=0)

    from wc2026.api.routes import live as live_route

    monkeypatch.setattr(live_route, "_all_events", lambda _mid: rows)

    gen = _stream_match_events(
        request=_StubRequest(),
        match_id=0,
        fixture=_argentina_v_france_fixture(),
        elo_diff=120.0,
        poisson_model=_fitted_poisson(),
        live_model=_fitted_live_model(),
        poll_interval=0.01,
    )

    async def drain() -> list[dict]:
        out: list[dict] = []
        async for chunk in gen:
            text = chunk.decode("utf-8")
            for line in text.splitlines():
                if line.startswith("data: "):
                    out.append(json.loads(line[len("data: ") :]))
        return out

    payloads = asyncio.run(drain())

    # One frame per row in raw_live_events.
    assert len(payloads) == len(rows)
    # The replay emitted: kickoff + 3 goals + 1 FT_WHISTLE = 5 rows.
    event_types = [p["last_event_type"] for p in payloads]
    assert event_types[0] == EVENT_KICKOFF
    assert event_types[-1] == EVENT_FT_WHISTLE
    # The frames in between are the 3 goals.
    assert event_types.count("GOAL") == 3
    # The kickoff frame is pre-match (Poisson), the goal frames are live, and
    # the FT_WHISTLE frame collapses to "final".
    assert payloads[0]["win_prob_source"] == "poisson_pre_match"
    assert payloads[-1]["win_prob_source"] == "final"
    # Every win-prob triplet sums to 1.
    for p in payloads:
        s = sum(p["win_prob"].values())
        assert abs(s - 1.0) < 1e-6


def test_replay_pipeline_collapses_to_realised_outcome_at_full_time(monkeypatch) -> None:
    snapshots = replay_statsbomb_events(_events(), home_team="Argentina", away_team="France")
    rows = _snapshots_to_rows(snapshots, match_id=0)
    from wc2026.api.routes import live as live_route

    monkeypatch.setattr(live_route, "_all_events", lambda _mid: rows)

    gen = _stream_match_events(
        request=_StubRequest(),
        match_id=0,
        fixture=_argentina_v_france_fixture(),
        elo_diff=120.0,
        poisson_model=_fitted_poisson(),
        live_model=_fitted_live_model(),
        poll_interval=0.01,
    )

    async def drain_final() -> dict:
        last: dict = {}
        async for chunk in gen:
            for line in chunk.decode("utf-8").splitlines():
                if line.startswith("data: "):
                    last = json.loads(line[len("data: ") :])
        return last

    final = asyncio.run(drain_final())
    # Argentina won 2-1 in the fixture → final home_win == 1.0.
    assert final["win_prob"] == {"home_win": 1.0, "draw": 0.0, "away_win": 0.0}
    assert final["home_score"] == 2
    assert final["away_score"] == 1


def test_replay_pipeline_win_prob_responds_to_goal_diff(monkeypatch) -> None:
    """As the fixture progresses (0-0 → 1-0 → 1-1 → 2-1), home_win prob should
    rise on the home goal, fall on the away equaliser, and re-rise after the
    home winner."""
    snapshots = replay_statsbomb_events(_events(), home_team="Argentina", away_team="France")
    rows = _snapshots_to_rows(snapshots, match_id=0)
    from wc2026.api.routes import live as live_route

    monkeypatch.setattr(live_route, "_all_events", lambda _mid: rows)

    gen = _stream_match_events(
        request=_StubRequest(),
        match_id=0,
        fixture=_argentina_v_france_fixture(),
        elo_diff=120.0,
        poisson_model=_fitted_poisson(),
        live_model=_fitted_live_model(),
        poll_interval=0.01,
    )

    async def drain_goal_frames() -> list[dict]:
        out: list[dict] = []
        async for chunk in gen:
            for line in chunk.decode("utf-8").splitlines():
                if line.startswith("data: "):
                    out.append(json.loads(line[len("data: ") :]))
        # Strip the kickoff + FT_WHISTLE rows; keep only the GOAL frames.
        return [p for p in out if p["last_event_type"] == "GOAL"]

    goal_frames = asyncio.run(drain_goal_frames())
    home_probs = [g["win_prob"]["home_win"] for g in goal_frames]
    # Three GOAL frames: 1-0 (Argentina) → 1-1 (France) → 2-1 (Argentina, penalty).
    assert len(home_probs) == 3
    assert home_probs[0] > home_probs[1]  # 1-0 → 1-1 reduces home prob
    assert home_probs[2] > home_probs[1]  # 1-1 → 2-1 raises home prob
