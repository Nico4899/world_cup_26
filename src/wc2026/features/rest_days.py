"""Days-of-rest features.

Rest days are the gap between a team's previous completed match and the
target match date. Long gaps are a known disadvantage in international
tournaments (jet-lag, lack of cohesion); tight turnarounds are a known
disadvantage at club level. The feature is signed in pair form:
``rest_days_diff = home_rest_days - away_rest_days``.

A team with no prior match in the corpus yields ``None`` — Phase 5's XGBoost
imputes the median, but feature-table consumers must handle the absence.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = ("date", "home_team", "away_team")


def _validate(matches: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS) - set(matches.columns)
    if missing:
        raise ValueError(f"rest_days input missing columns: {missing}")
    df = matches[list(REQUIRED_COLUMNS)].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


def last_match_date(
    matches: pd.DataFrame,
    *,
    team: str,
    as_of: date,
) -> date | None:
    """Most-recent match strictly before ``as_of`` involving ``team``.

    Returns ``None`` if the team has no prior matches in ``matches``.
    """
    df = _validate(matches)
    as_of_ts = pd.Timestamp(as_of)
    involved = df[((df["home_team"] == team) | (df["away_team"] == team)) & (df["date"] < as_of_ts)]
    if involved.empty:
        return None
    return involved["date"].max().date()


def rest_days(
    matches: pd.DataFrame,
    *,
    team: str,
    as_of: date,
) -> int | None:
    """Days between ``team``'s last match and ``as_of``. ``None`` if no history.

    A team that played yesterday and plays today gets ``rest_days == 1``.
    Same-day double-headers yield ``0``.
    """
    last = last_match_date(matches, team=team, as_of=as_of)
    if last is None:
        return None
    return (as_of - last).days


def rest_days_diff(
    matches: pd.DataFrame,
    *,
    home: str,
    away: str,
    as_of: date,
) -> int | None:
    """``home_rest_days - away_rest_days``. ``None`` if either side is missing."""
    h = rest_days(matches, team=home, as_of=as_of)
    a = rest_days(matches, team=away, as_of=as_of)
    if h is None or a is None:
        return None
    return h - a


__all__ = [
    "REQUIRED_COLUMNS",
    "last_match_date",
    "rest_days",
    "rest_days_diff",
]
