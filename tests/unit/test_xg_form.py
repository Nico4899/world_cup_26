"""Unit tests for the rolling xG-form feature module."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from wc2026.features.xg_form import (
    DEFAULT_WINDOW,
    compute_form_features,
    rolling_xg_form,
    xg_form_diff,
)


def _arg_history() -> pd.DataFrame:
    """Eight Argentina matches between 2024 and 2026."""
    return pd.DataFrame(
        {
            "match_date": pd.to_datetime(
                [
                    "2024-09-05",
                    "2024-11-15",
                    "2025-03-22",
                    "2025-06-10",
                    "2025-09-04",
                    "2025-11-13",
                    "2026-03-26",
                    "2026-05-15",
                ]
            ),
            "team": ["Argentina"] * 8,
            "xg_for": [1.2, 1.6, 0.9, 2.1, 1.5, 2.4, 2.7, 1.9],
            "xg_against": [0.8, 1.1, 1.4, 0.7, 0.9, 1.0, 0.4, 0.8],
        }
    )


def _two_team_history() -> pd.DataFrame:
    """Argentina + France, four matches each, well before 2026-06-11."""
    arg = _arg_history().iloc[3:].copy()  # last 5 Argentina matches
    fra = pd.DataFrame(
        {
            "match_date": pd.to_datetime(
                ["2025-09-04", "2025-11-13", "2026-03-22", "2026-04-08", "2026-05-30"]
            ),
            "team": ["France"] * 5,
            "xg_for": [2.3, 1.8, 1.1, 2.0, 1.6],
            "xg_against": [1.0, 0.9, 1.5, 1.2, 0.5],
        }
    )
    return pd.concat([arg, fra], ignore_index=True)


def test_rolling_xg_form_uses_last_n_matches_before_as_of() -> None:
    out = rolling_xg_form(
        _arg_history(),
        team="Argentina",
        as_of=pd.Timestamp("2026-06-11"),
        window=5,
    )
    # Last 5 Argentina matches (June 2025 onwards): mean xG_for over 2.1/1.5/2.4/2.7/1.9
    assert out["n_matches"] == 5
    assert math.isclose(out["xg_for_mean"], (2.1 + 1.5 + 2.4 + 2.7 + 1.9) / 5, abs_tol=1e-9)
    assert math.isclose(out["xg_against_mean"], (0.7 + 0.9 + 1.0 + 0.4 + 0.8) / 5, abs_tol=1e-9)


def test_rolling_xg_form_excludes_matches_on_or_after_as_of() -> None:
    """Matches dated == as_of must be excluded (strict <), to avoid leakage."""
    out = rolling_xg_form(
        _arg_history(),
        team="Argentina",
        as_of=pd.Timestamp("2025-09-04"),
        window=99,
    )
    # Matches strictly before 2025-09-04: 4 matches.
    assert out["n_matches"] == 4


def test_rolling_xg_form_handles_team_with_no_history() -> None:
    out = rolling_xg_form(
        _arg_history(),
        team="Atlantis",
        as_of=pd.Timestamp("2026-06-11"),
    )
    assert out["n_matches"] == 0
    assert math.isnan(out["xg_for_mean"])
    assert math.isnan(out["xg_against_mean"])


def test_rolling_xg_form_truncates_when_history_shorter_than_window() -> None:
    out = rolling_xg_form(
        _arg_history().iloc[-2:],  # only two matches available
        team="Argentina",
        as_of=pd.Timestamp("2026-06-11"),
        window=5,
    )
    assert out["n_matches"] == 2


def test_rolling_xg_form_validates_required_columns() -> None:
    bad = pd.DataFrame({"match_date": [pd.Timestamp("2026-01-01")], "team": ["X"]})
    with pytest.raises(ValueError, match="missing"):
        rolling_xg_form(bad, team="X", as_of=pd.Timestamp("2026-06-11"))


def test_compute_form_features_returns_one_row_per_team() -> None:
    form = compute_form_features(
        _two_team_history(),
        teams=["Argentina", "France"],
        as_of=pd.Timestamp("2026-06-11"),
        window=DEFAULT_WINDOW,
    )
    assert len(form) == 2
    assert set(form["team"]) == {"Argentina", "France"}


def test_xg_form_diff_positive_when_home_attacks_better() -> None:
    form = compute_form_features(
        _two_team_history(),
        teams=["Argentina", "France"],
        as_of=pd.Timestamp("2026-06-11"),
    )
    diff = xg_form_diff(form, home="Argentina", away="France")
    # Argentina recent xG_for - xG_against is higher than France's; diff should be > 0.
    assert diff > 0


def test_xg_form_diff_returns_nan_when_team_missing() -> None:
    form = compute_form_features(
        _two_team_history(),
        teams=["Argentina", "France"],
        as_of=pd.Timestamp("2026-06-11"),
    )
    diff = xg_form_diff(form, home="Argentina", away="Atlantis")
    assert math.isnan(diff)
