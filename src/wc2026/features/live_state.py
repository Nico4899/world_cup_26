"""Match-state replay for the live win-probability training set.

Given a per-event log (StatsBomb open data or our own ``raw_live_events``),
walk the events forward and emit one ``StateSnapshot`` per state-changing
event. The snapshots are then expanded into training rows for
``models.live_win_prob`` by tagging each with the eventual full-time outcome.

We also expose ``compute_current_state`` so the live API can summarise the
post-event state for one match in one call.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from wc2026.models.live_win_prob import minutes_remaining_from_minute
from wc2026.models.xgb_classifier import CLASS_AWAY, CLASS_DRAW, CLASS_HOME

# StatsBomb event-type names we care about.
TYPE_SHOT = "Shot"
TYPE_BAD_BEHAVIOUR = "Bad Behaviour"
TYPE_FOUL_COMMITTED = "Foul Committed"

# StatsBomb red-card outcome names (apply to both the bad_behaviour and the
# foul_committed event variants).
RED_CARD_NAMES: frozenset[str] = frozenset(
    {"Red Card", "Second Yellow", "Second Yellow Card"}
)


@dataclass(frozen=True)
class StateSnapshot:
    """Match state immediately after one event resolves."""

    minute: int
    period: int
    home_score: int
    away_score: int
    home_red_cards: int
    away_red_cards: int
    event_type: str = "KICKOFF"

    @property
    def goal_diff(self) -> int:
        return self.home_score - self.away_score

    @property
    def red_diff(self) -> int:
        return self.home_red_cards - self.away_red_cards


def _get(d: dict[str, Any] | None, *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _is_red_card(ev: dict[str, Any]) -> bool:
    ev_type = _get(ev, "type", "name")
    if ev_type == TYPE_BAD_BEHAVIOUR:
        return _get(ev, "bad_behaviour", "card", "name") in RED_CARD_NAMES
    if ev_type == TYPE_FOUL_COMMITTED:
        return _get(ev, "foul_committed", "card", "name") in RED_CARD_NAMES
    return False


def _is_goal(ev: dict[str, Any]) -> bool:
    if _get(ev, "type", "name") != TYPE_SHOT:
        return False
    return _get(ev, "shot", "outcome", "name") == "Goal"


def replay_statsbomb_events(
    events: Iterable[dict[str, Any]],
    *,
    home_team: str,
    away_team: str,
) -> list[StateSnapshot]:
    """Walk ``events`` in order, return one snapshot per state-changing event.

    The list always starts with a kickoff snapshot (minute=0, period=1, all
    zeros) so callers can plot a complete timeline including the pre-match
    state. Goals + red cards are the only events that change state for the
    live win-prob features.
    """
    snapshots: list[StateSnapshot] = [StateSnapshot(0, 1, 0, 0, 0, 0)]
    home_score = away_score = 0
    home_reds = away_reds = 0
    for ev in events:
        team = _get(ev, "team", "name") or ""
        event_type: str | None = None
        if _is_goal(ev):
            if team == home_team:
                home_score += 1
            elif team == away_team:
                away_score += 1
            event_type = "GOAL"
        elif _is_red_card(ev):
            if team == home_team:
                home_reds += 1
            elif team == away_team:
                away_reds += 1
            event_type = "RED_CARD"
        if event_type is None:
            continue
        snapshots.append(
            StateSnapshot(
                minute=int(ev.get("minute") or 0),
                period=int(ev.get("period") or 1),
                home_score=home_score,
                away_score=away_score,
                home_red_cards=home_reds,
                away_red_cards=away_reds,
                event_type=event_type,
            )
        )
    return snapshots


def outcome_label_from_final_score(home_score: int, away_score: int) -> int:
    """Map a full-time score to the H/D/A class label used by the live model."""
    if home_score > away_score:
        return CLASS_HOME
    if home_score < away_score:
        return CLASS_AWAY
    return CLASS_DRAW


def snapshots_to_training_rows(
    snapshots: list[StateSnapshot],
    *,
    elo_diff: float,
    final_home_score: int,
    final_away_score: int,
) -> list[dict[str, Any]]:
    """Expand a snapshot list into rows for ``LiveWinProbModel.fit``.

    Each row carries the four model features and the eventual outcome label.
    """
    label = outcome_label_from_final_score(final_home_score, final_away_score)
    return [
        {
            "elo_diff": float(elo_diff),
            "goal_diff": snap.goal_diff,
            "minutes_remaining": minutes_remaining_from_minute(snap.minute, snap.period),
            "red_diff": snap.red_diff,
            "label": label,
        }
        for snap in snapshots
    ]


def compute_current_state(snapshots: list[StateSnapshot]) -> StateSnapshot:
    """Return the most recent snapshot from a non-empty list."""
    if not snapshots:
        raise ValueError("snapshots is empty — replay must start with a kickoff state")
    return snapshots[-1]


__all__ = [
    "RED_CARD_NAMES",
    "StateSnapshot",
    "compute_current_state",
    "outcome_label_from_final_score",
    "replay_statsbomb_events",
    "snapshots_to_training_rows",
]
