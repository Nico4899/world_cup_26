"""Unit tests for the StatsBomb-events → state replay helper."""

from __future__ import annotations

import json
from pathlib import Path

from wc2026.features.live_state import (
    StateSnapshot,
    compute_current_state,
    outcome_label_from_final_score,
    replay_statsbomb_events,
    snapshots_to_training_rows,
)
from wc2026.models.xgb_classifier import CLASS_AWAY, CLASS_DRAW, CLASS_HOME

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"


def _events() -> list[dict]:
    return json.loads(
        (FIXTURE_DIR / "statsbomb_events_sample.json").read_text(encoding="utf-8")
    )


def test_replay_starts_with_kickoff_snapshot() -> None:
    snapshots = replay_statsbomb_events(_events(), home_team="Argentina", away_team="France")
    first = snapshots[0]
    assert first == StateSnapshot(0, 1, 0, 0, 0, 0)


def test_replay_emits_one_snapshot_per_goal() -> None:
    """The fixture has 3 goals: Messi 23' + Mbappé 80' + Messi 117'."""
    snapshots = replay_statsbomb_events(_events(), home_team="Argentina", away_team="France")
    # Kickoff + 3 goals = 4 snapshots.
    assert len(snapshots) == 4


def test_replay_accumulates_scores() -> None:
    snapshots = replay_statsbomb_events(_events(), home_team="Argentina", away_team="France")
    # Final snapshot: Argentina 2-1 France (penalty by Messi).
    final = snapshots[-1]
    assert final.home_score == 2
    assert final.away_score == 1


def test_replay_ignores_non_state_events() -> None:
    """The fixture also contains a non-goal Shot (saved) and a Pass — both should
    leave state unchanged."""
    events_with_pass = _events()
    snapshots = replay_statsbomb_events(events_with_pass, home_team="Argentina", away_team="France")
    # Only kickoff + 3 goals; the saved shot and the pass don't emit snapshots.
    assert len(snapshots) == 4


def test_replay_tracks_red_cards() -> None:
    events = [
        {
            "type": {"name": "Bad Behaviour"},
            "team": {"name": "Argentina"},
            "minute": 30,
            "period": 1,
            "bad_behaviour": {"card": {"name": "Red Card"}},
        },
        {
            "type": {"name": "Foul Committed"},
            "team": {"name": "France"},
            "minute": 55,
            "period": 2,
            "foul_committed": {"card": {"name": "Second Yellow"}},
        },
    ]
    snapshots = replay_statsbomb_events(events, home_team="Argentina", away_team="France")
    # Kickoff + 2 reds = 3 snapshots.
    assert len(snapshots) == 3
    assert snapshots[1].home_red_cards == 1
    assert snapshots[1].away_red_cards == 0
    assert snapshots[2].home_red_cards == 1
    assert snapshots[2].away_red_cards == 1


def test_replay_emits_goal_diff_and_red_diff() -> None:
    """Properties on StateSnapshot should compute from the raw counts."""
    snapshots = replay_statsbomb_events(_events(), home_team="Argentina", away_team="France")
    # After the Mbappé 80' goal: Argentina 1-1 France → goal_diff = 0
    one_one = snapshots[2]
    assert one_one.goal_diff == 0


def test_outcome_label_from_final_score() -> None:
    assert outcome_label_from_final_score(2, 1) == CLASS_HOME
    assert outcome_label_from_final_score(1, 1) == CLASS_DRAW
    assert outcome_label_from_final_score(0, 2) == CLASS_AWAY


def test_snapshots_to_training_rows_carries_label_and_features() -> None:
    snapshots = replay_statsbomb_events(_events(), home_team="Argentina", away_team="France")
    rows = snapshots_to_training_rows(
        snapshots, elo_diff=104.0, final_home_score=2, final_away_score=1
    )
    assert len(rows) == 4  # one per snapshot
    assert {row["label"] for row in rows} == {CLASS_HOME}  # final outcome was a home win
    # First row is kickoff: goal_diff=0, minutes_remaining=90, red_diff=0
    first = rows[0]
    assert first["goal_diff"] == 0
    assert first["minutes_remaining"] == 90
    assert first["red_diff"] == 0
    assert first["elo_diff"] == 104.0


def test_snapshots_to_training_rows_minutes_remaining_falls_with_clock() -> None:
    snapshots = replay_statsbomb_events(_events(), home_team="Argentina", away_team="France")
    rows = snapshots_to_training_rows(
        snapshots, elo_diff=0.0, final_home_score=2, final_away_score=1
    )
    minutes_remaining = [row["minutes_remaining"] for row in rows]
    # Kickoff snapshot is 90; later snapshots all happen after kickoff so they
    # should be ≤ 90 (clamped at 0 for the ET snapshot in this fixture).
    assert minutes_remaining[0] == 90
    assert all(m <= 90 for m in minutes_remaining)


def test_compute_current_state_returns_latest_snapshot() -> None:
    snapshots = replay_statsbomb_events(_events(), home_team="Argentina", away_team="France")
    current = compute_current_state(snapshots)
    assert current is snapshots[-1]


def test_compute_current_state_raises_on_empty() -> None:
    import pytest

    with pytest.raises(ValueError, match="empty"):
        compute_current_state([])
