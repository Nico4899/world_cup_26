"""Travel-fatigue features.

International tournaments concentrate matches into a tight window with
significant cross-country travel — Toronto -> Monterrey is ~3 100 km, the
kind of jet-lag windfall the 1-yard altitude penalty in our v1 model
already gestures toward but doesn't quantify. The feature pair this
module produces:

* ``team_travel_km(matches, team, as_of, current_lat, current_lon)`` —
  great-circle distance from the team's PREVIOUS match venue to the
  upcoming match venue. ``None`` if the team has no prior recorded
  match in ``matches`` or the row doesn't carry venue coordinates.
* ``travel_km_diff`` (defined in :mod:`wc2026.features.build_match_features`)
  — ``home_travel_km - away_travel_km``. Positive = the home side
  travelled more than the away side; negative = away.

The math primitive (:func:`great_circle_km`) is pure haversine, no
external dependency. Coordinates accepted in decimal degrees.

The source DataFrame is expected to have ``date, home_team, away_team,
home_lat, home_lon`` columns. For WC 2026 we can derive ``home_lat`` /
``home_lon`` per fixture from the static host-cities table. For
pre-tournament games, the venue coordinates aren't on file — the
feature naturally degrades to ``None``, and XGB's hist tree-method
handles the NaN natively.
"""

from __future__ import annotations

import math
from datetime import date

import pandas as pd

EARTH_RADIUS_KM: float = 6371.0
REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "home_team",
    "away_team",
    "home_lat",
    "home_lon",
)


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance between two lat/lon points in kilometres.

    >>> round(great_circle_km(40.7128, -74.0060, 51.5074, -0.1278), 0)  # NYC -> LON
    5570.0
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    c = 2.0 * math.asin(min(1.0, math.sqrt(a)))
    return EARTH_RADIUS_KM * c


def _validate(matches: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS) - set(matches.columns)
    if missing:
        raise ValueError(f"travel input missing columns: {sorted(missing)}")
    df = matches[list(REQUIRED_COLUMNS)].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


def team_travel_km(
    matches: pd.DataFrame,
    *,
    team: str,
    as_of: date,
    current_lat: float,
    current_lon: float,
) -> float | None:
    """Great-circle km from ``team``'s previous match venue to (current_*).

    Looks for the most-recent row strictly before ``as_of`` where the team
    appears (either side). Returns ``None`` when:

    * the team has no prior row in ``matches``;
    * the prior row's ``home_lat`` / ``home_lon`` is NaN.

    NOTE — for now we only carry the home-side venue coordinates on a
    fixture row. This is exact for the home-team match-up but undercounts
    AWAY teams' travel by treating the away team as if it played "at the
    home venue" the day before. For an *international* tournament where
    nearly every match is at a neutral venue, that simplification is
    accurate; for friendlies and qualifiers at a non-neutral venue the
    away team's travel is approximated by the venue they're playing at
    next, which is the most useful proxy in the absence of squad-by-squad
    pre-camp data.
    """
    df = _validate(matches)
    as_of_ts = pd.Timestamp(as_of)
    involved = df[
        ((df["home_team"] == team) | (df["away_team"] == team))
        & (df["date"] < as_of_ts)
    ]
    if involved.empty:
        return None
    last = involved.loc[involved["date"].idxmax()]
    prev_lat = last["home_lat"]
    prev_lon = last["home_lon"]
    if pd.isna(prev_lat) or pd.isna(prev_lon):
        return None
    return great_circle_km(float(prev_lat), float(prev_lon), current_lat, current_lon)


def travel_km_diff(
    matches: pd.DataFrame,
    *,
    home: str,
    away: str,
    as_of: date,
    current_lat: float,
    current_lon: float,
) -> float | None:
    """``home_travel_km - away_travel_km``. ``None`` if either side is missing."""
    h = team_travel_km(
        matches,
        team=home,
        as_of=as_of,
        current_lat=current_lat,
        current_lon=current_lon,
    )
    a = team_travel_km(
        matches,
        team=away,
        as_of=as_of,
        current_lat=current_lat,
        current_lon=current_lon,
    )
    if h is None or a is None:
        return None
    return h - a


__all__ = [
    "EARTH_RADIUS_KM",
    "REQUIRED_COLUMNS",
    "great_circle_km",
    "team_travel_km",
    "travel_km_diff",
]
