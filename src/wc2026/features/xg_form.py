"""Rolling per-team xG-form features.

Input: a per-(match, team) xG DataFrame (the output of
``ingest.statsbomb_open.aggregate_match_xg`` or an equivalent FBref-derived
frame). Output: per-team rolling averages of ``xg_for`` and ``xg_against``
over the last N matches before a given reference date.

These are inputs to Phase 5's XGBoost classifier, where the difference between
the two teams' forms becomes a single feature (``xg_form_diff``).
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "match_date",
    "team",
    "xg_for",
    "xg_against",
)

DEFAULT_WINDOW = 5


def _validate(team_matches: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS) - set(team_matches.columns)
    if missing:
        raise ValueError(f"xG-form input is missing required columns: {missing}")
    df = team_matches.copy()
    df["match_date"] = pd.to_datetime(df["match_date"])
    return df


def rolling_xg_form(
    team_matches: pd.DataFrame,
    *,
    team: str,
    as_of: pd.Timestamp,
    window: int = DEFAULT_WINDOW,
) -> dict[str, float | int]:
    """Mean xG_for / xG_against over ``team``'s last ``window`` matches before ``as_of``.

    Returns:
        {
          "team": team,
          "as_of": as_of,
          "n_matches": <how many of the window were available>,
          "xg_for_mean": float | NaN,
          "xg_against_mean": float | NaN,
        }

    NaN means the team had zero matches in the corpus before ``as_of``.
    """
    df = _validate(team_matches)
    as_of_ts = pd.Timestamp(as_of)
    history = df[(df["team"] == team) & (df["match_date"] < as_of_ts)].copy()
    history = history.sort_values("match_date", ascending=False).head(window)
    if history.empty:
        return {
            "team": team,
            "as_of": as_of_ts,
            "n_matches": 0,
            "xg_for_mean": float("nan"),
            "xg_against_mean": float("nan"),
        }
    return {
        "team": team,
        "as_of": as_of_ts,
        "n_matches": len(history),
        "xg_for_mean": float(history["xg_for"].mean()),
        "xg_against_mean": float(history["xg_against"].mean()),
    }


def compute_form_features(
    team_matches: pd.DataFrame,
    *,
    teams: Iterable[str],
    as_of: pd.Timestamp,
    window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """Vectorised wrapper: apply ``rolling_xg_form`` to every team in ``teams``.

    Returns a DataFrame with columns
    ``team, as_of, n_matches, xg_for_mean, xg_against_mean``.
    """
    rows = [rolling_xg_form(team_matches, team=team, as_of=as_of, window=window) for team in teams]
    return pd.DataFrame(rows)


def xg_form_diff(form: pd.DataFrame, *, home: str, away: str) -> float:
    """Combined "home advantage" feature from a form DataFrame.

    Returns ``(home.xg_for - home.xg_against) - (away.xg_for - away.xg_against)``.
    NaN if either team's form row is missing or has zero matches.
    """
    indexed = form.set_index("team")
    if home not in indexed.index or away not in indexed.index:
        return float("nan")
    h, a = indexed.loc[home], indexed.loc[away]
    if int(h.get("n_matches", 0)) == 0 or int(a.get("n_matches", 0)) == 0:
        return float("nan")
    h_diff = float(h["xg_for_mean"]) - float(h["xg_against_mean"])
    a_diff = float(a["xg_for_mean"]) - float(a["xg_against_mean"])
    if math.isnan(h_diff) or math.isnan(a_diff):
        return float("nan")
    return h_diff - a_diff


__all__ = [
    "DEFAULT_WINDOW",
    "REQUIRED_COLUMNS",
    "compute_form_features",
    "rolling_xg_form",
    "xg_form_diff",
]
