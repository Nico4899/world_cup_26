"""Tests for the great-circle travel-fatigue helpers."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from wc2026.features import travel


def test_great_circle_zero_distance_for_same_point() -> None:
    assert travel.great_circle_km(40.0, -73.0, 40.0, -73.0) == pytest.approx(0.0, abs=1e-9)


def test_great_circle_nyc_to_london_known_value() -> None:
    # Classic textbook example: ~5 570 km between JFK (40.71, -74.01) and LHR (51.51, -0.13).
    d = travel.great_circle_km(40.7128, -74.0060, 51.5074, -0.1278)
    assert d == pytest.approx(5570, abs=20)


def test_great_circle_is_symmetric() -> None:
    a = travel.great_circle_km(33.75, -84.40, 25.96, -80.24)  # Atlanta -> Miami
    b = travel.great_circle_km(25.96, -80.24, 33.75, -84.40)
    assert a == pytest.approx(b, rel=1e-9)


def test_great_circle_antipodes_close_to_half_circumference() -> None:
    # (0, 0) and (0, 180) are antipodal; great-circle distance ~= pi * R.
    d = travel.great_circle_km(0.0, 0.0, 0.0, 180.0)
    # Pi * 6371 = ~20 015 km.
    assert d == pytest.approx(20015.0, rel=1e-3)


# --- team_travel_km ----------------------------------------------------------


def _history() -> pd.DataFrame:
    """Two-team, three-match history with venue coordinates per row."""
    return pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2026-06-11"),
                pd.Timestamp("2026-06-14"),
                pd.Timestamp("2026-06-18"),
            ],
            "home_team": ["Argentina", "Brazil", "Argentina"],
            "away_team": ["Saudi Arabia", "Switzerland", "Mexico"],
            # Mexico City -> Dallas -> Atlanta
            "home_lat": [19.3029, 32.7473, 33.7553],
            "home_lon": [-99.1505, -97.0945, -84.4006],
        }
    )


def test_team_travel_km_uses_previous_match_venue() -> None:
    """Argentina played at Mexico City on 06-11 + Atlanta on 06-18;
    asking about an 06-22 match at Miami should measure Atlanta->Miami."""
    km = travel.team_travel_km(
        _history(),
        team="Argentina",
        as_of=date(2026, 6, 22),
        current_lat=25.958,
        current_lon=-80.2389,  # Miami
    )
    # Atlanta -> Miami is approximately 950-1000 km; loose tolerance
    # because intercity distances aren't a precision contract.
    assert km is not None
    assert 900 < km < 1050


def test_team_travel_km_returns_none_for_team_with_no_history() -> None:
    km = travel.team_travel_km(
        _history(),
        team="Norway",  # never appears
        as_of=date(2026, 6, 22),
        current_lat=25.958,
        current_lon=-80.2389,
    )
    assert km is None


def test_team_travel_km_returns_none_when_prior_row_lacks_coords() -> None:
    df = _history()
    df.loc[df.index[2], "home_lat"] = float("nan")
    km = travel.team_travel_km(
        df,
        team="Argentina",
        as_of=date(2026, 6, 22),
        current_lat=25.958,
        current_lon=-80.2389,
    )
    assert km is None


def test_team_travel_km_ignores_rows_at_or_after_as_of() -> None:
    """A team that 'plays at as_of' must not count that match as the prior."""
    df = _history()
    df.loc[df.index[2], "date"] = pd.Timestamp("2026-06-22")
    km = travel.team_travel_km(
        df,
        team="Argentina",
        as_of=date(2026, 6, 22),
        current_lat=25.958,
        current_lon=-80.2389,
    )
    # Falls back to Mexico City -> Miami (Argentina's 06-11 match) which
    # is roughly 2050-2150 km.
    assert km is not None
    assert 2000 < km < 2200


# --- travel_km_diff ---------------------------------------------------------


def test_travel_km_diff_is_signed_home_minus_away() -> None:
    diff = travel.travel_km_diff(
        _history(),
        home="Argentina",  # last at Atlanta -> Miami (smaller hop)
        away="Brazil",  # last at Dallas -> Miami (bigger hop)
        as_of=date(2026, 6, 22),
        current_lat=25.958,
        current_lon=-80.2389,
    )
    assert diff is not None
    # home (smaller) - away (bigger) → negative.
    assert diff < 0
    # The gap is several hundred km — sanity-bound it without pinning a
    # precise value.
    assert -1200 < diff < -500


def test_travel_km_diff_none_when_either_side_missing() -> None:
    assert (
        travel.travel_km_diff(
            _history(),
            home="Argentina",
            away="Norway",
            as_of=date(2026, 6, 22),
            current_lat=25.958,
            current_lon=-80.2389,
        )
        is None
    )


def test_validate_rejects_missing_columns() -> None:
    bad = pd.DataFrame({"date": [], "home_team": [], "away_team": []})
    with pytest.raises(ValueError, match="missing columns"):
        travel.team_travel_km(
            bad,
            team="X",
            as_of=date(2026, 6, 11),
            current_lat=0.0,
            current_lon=0.0,
        )
