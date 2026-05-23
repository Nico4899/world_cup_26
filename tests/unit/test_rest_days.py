"""Unit tests for the rest-days feature module."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from wc2026.features.rest_days import (
    last_match_date,
    rest_days,
    rest_days_diff,
)


def _matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-05-10",  # Argentina vs Bolivia
                    "2026-05-30",  # France vs Spain
                    "2026-06-05",  # Argentina vs Mexico
                    "2026-06-05",  # France vs USA
                    "2026-06-10",  # Brazil vs Italy (Argentina/France not involved)
                ]
            ),
            "home_team": ["Argentina", "France", "Argentina", "France", "Brazil"],
            "away_team": ["Bolivia", "Spain", "Mexico", "USA", "Italy"],
        }
    )


def test_last_match_date_picks_most_recent_strictly_before_as_of() -> None:
    last = last_match_date(_matches(), team="Argentina", as_of=date(2026, 6, 11))
    assert last == date(2026, 6, 5)


def test_last_match_date_returns_none_when_no_history() -> None:
    assert last_match_date(_matches(), team="Atlantis", as_of=date(2026, 6, 11)) is None


def test_last_match_date_excludes_match_on_as_of_itself() -> None:
    """Same-day match must NOT count as 'last match before today'."""
    last = last_match_date(_matches(), team="Argentina", as_of=date(2026, 6, 5))
    # Strictly < 2026-06-05 → 2026-05-10.
    assert last == date(2026, 5, 10)


def test_rest_days_counts_calendar_days() -> None:
    days = rest_days(_matches(), team="Argentina", as_of=date(2026, 6, 11))
    assert days == 6


def test_rest_days_zero_for_same_day_doubleheader() -> None:
    """If the team plays again on the same day as a previous match, rest=0 once
    the previous match has happened — but we measure strictly < as_of, so the
    same-day case (history exactly == as_of) returns None. Asymmetric on purpose;
    a real same-day double-header would have one of the matches dated +1."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-10", "2026-06-11"]),
            "home_team": ["Argentina", "Argentina"],
            "away_team": ["A", "B"],
        }
    )
    assert rest_days(df, team="Argentina", as_of=date(2026, 6, 11)) == 1


def test_rest_days_diff_positive_when_home_better_rested() -> None:
    # As of 2026-06-12: Argentina last played 2026-06-05 (7 days);
    # France last played 2026-06-05 (7 days). Diff = 0.
    diff = rest_days_diff(
        _matches(), home="Argentina", away="France", as_of=date(2026, 6, 12)
    )
    assert diff == 0


def test_rest_days_diff_picks_up_asymmetry() -> None:
    # France played 2026-05-30 + 2026-06-05; Argentina played 2026-05-10 + 2026-06-05.
    # As of 2026-06-06: both played 2026-06-05 → diff = 0.
    # As of 2026-06-04 (before June 5 matches): France = 2026-05-30 (5d), Argentina = 2026-05-10 (25d).
    diff = rest_days_diff(
        _matches(), home="Argentina", away="France", as_of=date(2026, 6, 4)
    )
    assert diff == 25 - 5


def test_rest_days_diff_returns_none_when_either_team_missing() -> None:
    diff = rest_days_diff(
        _matches(), home="Argentina", away="Atlantis", as_of=date(2026, 6, 11)
    )
    assert diff is None


def test_rest_days_validates_required_columns() -> None:
    bad = pd.DataFrame({"date": [pd.Timestamp("2026-01-01")], "team": ["X"]})
    with pytest.raises(ValueError, match="missing"):
        last_match_date(bad, team="X", as_of=date(2026, 6, 11))
