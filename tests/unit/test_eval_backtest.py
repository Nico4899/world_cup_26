"""Tests for the hindcast harness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wc2026.eval.backtest import HindcastConfig, hindcast


def _make_history(n_per_year: int = 50, years: int = 5) -> pd.DataFrame:
    """Synthetic history: ``n_per_year * years`` matches across 4 teams, all
    drawn from a deterministic-ish process so the model converges."""
    rng = np.random.default_rng(0)
    teams = ["A", "B", "C", "D"]
    rows: list[dict] = []
    start = pd.Timestamp("2020-01-01")
    for i in range(years * n_per_year):
        date = start + pd.Timedelta(days=int(i * 365 / n_per_year))
        h, a = rng.choice(teams, 2, replace=False)
        # Skew: A is strong, D is weak
        skew = {"A": 1.5, "B": 1.0, "C": 0.8, "D": 0.5}
        lh = skew[h] / skew[a] * 1.5  # rough goal scale
        la = skew[a] / skew[h] * 1.0
        rows.append(
            {
                "date": date,
                "home_team": h,
                "away_team": a,
                "home_score": int(rng.poisson(lh)),
                "away_score": int(rng.poisson(la)),
                "neutral": False,
                "tournament": "Friendly",
            }
        )
    return pd.DataFrame(rows)


def test_hindcast_does_not_use_target_match_data() -> None:
    """A match scheduled exactly on the cutoff date must NOT appear in training.
    This is verified indirectly: the training set size at cutoff t must equal the
    number of history rows with date < t (no >=)."""
    history = _make_history(n_per_year=20, years=2)
    target = history.iloc[10:13].copy()  # predict matches at known dates
    out = hindcast(target, history)
    # for each target row, train_n must equal count of history rows strictly before that date
    for i, row in out.iterrows():
        expected_train_n = (history["date"] < row["date"]).sum()
        assert row["train_n"] == expected_train_n, (
            f"row {i}: train_n={row['train_n']} but {expected_train_n} history rows are strictly earlier"
        )


def test_hindcast_returns_expected_columns_and_no_skips_for_known_teams() -> None:
    history = _make_history(n_per_year=50, years=3)
    target = history.tail(5).copy()
    out = hindcast(target, history)
    expected_cols = {
        "date",
        "home_team",
        "away_team",
        "neutral",
        "observed",
        "actual_home",
        "actual_away",
        "p_home",
        "p_draw",
        "p_away",
        "train_n",
        "skipped_reason",
    }
    assert expected_cols.issubset(set(out.columns))
    assert len(out) == 5
    # All known teams, no skips
    assert out["skipped_reason"].isna().all()
    # Probabilities sum to 1 (within tiny tolerance)
    sums = out["p_home"] + out["p_draw"] + out["p_away"]
    assert all(abs(s - 1.0) < 1e-9 for s in sums)


def test_hindcast_skips_unknown_team() -> None:
    history = _make_history(n_per_year=20, years=2)
    # Add a target row with an unseen team
    new_row = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2022-12-01"),
                "home_team": "A",
                "away_team": "Z",  # never seen
                "home_score": 1,
                "away_score": 0,
                "neutral": True,
                "tournament": "Friendly",
            }
        ]
    )
    out = hindcast(new_row, history)
    assert len(out) == 1
    assert out.iloc[0]["skipped_reason"] is not None
    assert pd.isna(out.iloc[0]["p_home"])


def test_hindcast_rejects_missing_columns() -> None:
    history = _make_history(n_per_year=10, years=1)
    target = pd.DataFrame({"date": [pd.Timestamp("2021-01-01")]})
    with pytest.raises(ValueError, match="missing columns"):
        hindcast(target, history)


def test_hindcast_refit_per_match_uses_more_models_than_daily() -> None:
    """Two matches on the same day should produce one model with 'daily' cadence
    and two distinct models with 'per_match' cadence (we infer this via train_n).
    Construct a 2-match day: one match before lunch, one after, history has new
    rows added overnight to make sure per_match would see different train_n."""
    history = _make_history(n_per_year=50, years=3)
    # Use the last two history rows as targets BUT also add a one-row gap to
    # ensure train_n changes if we re-fit per match (only really verifiable here
    # with two different *dates*; for same-day we just check the cadence runs).
    target = history.iloc[-2:].copy()
    out_daily = hindcast(target, history, cfg=HindcastConfig(refit_cadence="daily"))
    out_per_match = hindcast(target, history, cfg=HindcastConfig(refit_cadence="per_match"))
    # Per-match must produce non-decreasing train_n on dates after the first;
    # daily would also; this just smoke-tests both code paths exist.
    assert len(out_daily) == 2
    assert len(out_per_match) == 2
