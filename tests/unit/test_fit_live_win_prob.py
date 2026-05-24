"""Unit tests for scripts/fit_live_win_prob.py.

We use synthetic state-snapshot rows so the test doesn't need any StatsBomb
data on disk; the StatsBomb-corpus path is exercised by the
``test_replay_integration`` test in Phase 6.8.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.fit_live_win_prob as f
from wc2026.models.live_win_prob import LiveWinProbModel


def _synthetic_rows(n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    elo_diff = rng.normal(0.0, 80.0, size=n)
    goal_diff = rng.integers(-3, 4, size=n)
    minutes_remaining = rng.integers(0, 90, size=n)
    red_diff = rng.choice([-1, 0, 1], size=n, p=[0.05, 0.9, 0.05])
    logits = 0.01 * elo_diff + 1.4 * goal_diff - 0.4 * red_diff
    noise = rng.normal(0, 1.0, size=n)
    labels = np.where(
        logits + noise > 0.8, 0, np.where(logits + noise < -0.8, 2, 1)
    ).astype(int)
    return pd.DataFrame(
        {
            "elo_diff": elo_diff,
            "goal_diff": goal_diff,
            "minutes_remaining": minutes_remaining,
            "red_diff": red_diff,
            "label": labels,
        }
    )


def test_fit_from_rows_returns_model_with_correct_classes() -> None:
    model = f.fit_from_rows(_synthetic_rows())
    assert len(model.intercepts) == 3
    assert all(len(row) == 4 for row in model.coefficients)


def test_fit_from_rows_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="no rows"):
        f.fit_from_rows(pd.DataFrame(columns=["elo_diff", "label"]))


def test_fit_and_save_writes_artefact(tmp_path) -> None:
    out = f.fit_and_save(rows=_synthetic_rows(), artifact_path=tmp_path / "model.json")
    assert out == tmp_path / "model.json"
    reloaded = LiveWinProbModel.load(out)
    assert len(reloaded.intercepts) == 3


def test_build_training_rows_returns_empty_when_no_disk_corpus(tmp_path) -> None:
    """A fresh, never-populated statsbomb cache yields an empty rows df."""
    df = f.build_training_rows(target_dir=tmp_path)
    assert df.empty
    assert {"elo_diff", "goal_diff", "minutes_remaining", "red_diff", "label"}.issubset(
        df.columns
    )


def test_build_training_rows_walks_replay_for_one_match(monkeypatch, tmp_path) -> None:
    """End-to-end with stubbed HTTP: one match → ~4 rows (kickoff + 3 goals)."""
    import json
    from pathlib import Path

    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    events = json.loads(
        (fixtures_dir / "statsbomb_events_sample.json").read_text(encoding="utf-8")
    )

    # Lay out a synthetic shots.parquet so _resolve_competition_seasons finds it.
    (tmp_path / "43" / "106").mkdir(parents=True)
    (tmp_path / "43" / "106" / "shots.parquet").write_bytes(b"")

    def fake_fetch_matches(comp, season, **_):
        assert comp == 43 and season == 106
        return [
            {
                "match_id": 1,
                "home_team": {"home_team_name": "Argentina"},
                "away_team": {"away_team_name": "France"},
                "home_score": 2,
                "away_score": 1,
            }
        ]

    def fake_fetch_events(match_id, **_):
        assert match_id == 1
        return events

    monkeypatch.setattr(f, "fetch_matches", fake_fetch_matches)
    monkeypatch.setattr(f, "fetch_match_events", fake_fetch_events)
    df = f.build_training_rows(target_dir=tmp_path, elo_df=None)
    # kickoff + 3 goals = 4 rows
    assert len(df) == 4
    # all rows carry the final-outcome label (home win) since Argentina won.
    assert set(df["label"]) == {0}


def test_build_training_rows_respects_max_matches(monkeypatch, tmp_path) -> None:
    """``max_matches`` caps the number of contributing matches."""
    import json
    from pathlib import Path

    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    events = json.loads(
        (fixtures_dir / "statsbomb_events_sample.json").read_text(encoding="utf-8")
    )
    (tmp_path / "43" / "106").mkdir(parents=True)
    (tmp_path / "43" / "106" / "shots.parquet").write_bytes(b"")

    monkeypatch.setattr(
        f,
        "fetch_matches",
        lambda *_a, **_k: [
            {
                "match_id": i,
                "home_team": {"home_team_name": "Argentina"},
                "away_team": {"away_team_name": "France"},
                "home_score": 2,
                "away_score": 1,
            }
            for i in range(1, 6)
        ],
    )
    monkeypatch.setattr(f, "fetch_match_events", lambda *_a, **_k: events)
    df = f.build_training_rows(target_dir=tmp_path, elo_df=None, max_matches=2)
    # 2 matches × 4 snapshots each = 8 rows
    assert len(df) == 8
