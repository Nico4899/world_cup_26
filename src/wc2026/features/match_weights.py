"""Match weighting for the Poisson fit.

We multiply two factors:

1. **Time decay** — exponential with a configurable half-life.
   weight_time = 0.5 ** (age_days / half_life_days)

2. **Match importance** — the World Football Elo K-factor schedule, which assigns
   higher influence to high-stakes tournaments. Source: eloratings.net "About"
   page (last reviewed 2026-05-23).

       K = 60  World Cup finals
       K = 50  Continental championship finals; Confederations Cup
       K = 40  WC/continental qualifiers; major intercontinental tournaments;
               Nations Leagues
       K = 30  All other tournaments
       K = 20  Friendly matches

Match importance is not normalised — absolute scale of MLE weights does not affect
the optimum; only ratios do.

For the half-life: the often-cited 390-day half-life of Ley, Van de Wiele &
Van Eetvelde (2019) was fitted on the English Premier League, where teams play
~38 matches per season. International teams play ~10 matches per year, so this
value is almost certainly too short. The right approach is to sweep the half-life
on the WC 2022 backtest (Stage 0.6); a value in the 600-1500 day range is the
expected ballpark.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd

# K-factor for the World Cup final stage. Use this constant as the reference scale.
K_WORLD_CUP_FINAL: int = 60
K_CONTINENTAL_FINAL: int = 50
K_QUALIFIER_OR_MAJOR: int = 40
K_OTHER_TOURNAMENT: int = 30
K_FRIENDLY: int = 20

# Names that map to K=50 (continental championship finals + Confederations Cup).
# Names that begin with these strings PLUS lack "qualification"/"qualifier" qualify.
_CONTINENTAL_FINAL_PREFIXES: tuple[str, ...] = (
    "UEFA Euro",
    "Copa América",
    "Copa America",  # variant without accent, just in case
    "African Cup of Nations",
    "Africa Cup of Nations",
    "AFC Asian Cup",
    "CONCACAF Gold Cup",
    "Gold Cup",
    "OFC Nations Cup",
    "Oceania Nations Cup",
    "FIFA Confederations Cup",
    "Confederations Cup",
)

_QUALIFIER_TOKENS: tuple[str, ...] = ("qualification", "qualifier", "Qualification", "Qualifier")


def match_importance_weight(tournament: str | None) -> int:  # noqa: PLR0911 — classifier with one branch per K-bucket reads better than nested elif
    """Return the World Football Elo K-factor for a tournament name.

    Unknown / None tournament → K=30 (the "other tournament" bucket).
    """
    if not tournament:
        return K_OTHER_TOURNAMENT
    if tournament == "FIFA World Cup":
        return K_WORLD_CUP_FINAL
    if tournament == "Friendly":
        return K_FRIENDLY
    if any(tok in tournament for tok in _QUALIFIER_TOKENS):
        return K_QUALIFIER_OR_MAJOR
    if "Nations League" in tournament:
        return K_QUALIFIER_OR_MAJOR
    if any(tournament.startswith(p) for p in _CONTINENTAL_FINAL_PREFIXES):
        return K_CONTINENTAL_FINAL
    return K_OTHER_TOURNAMENT


def time_decay_weight(
    match_date: pd.Series | pd.Timestamp | datetime,
    ref_date: pd.Timestamp | datetime,
    half_life_days: float,
) -> pd.Series | float:
    """Return exp-decay weight 0.5 ** (age_days / H).

    Negative age (future match relative to ref_date) is clamped to 0, so future
    matches get weight 1.0. This is intentional: future matches in the training set
    are an upstream bug, not an input.
    """
    if half_life_days <= 0:
        raise ValueError(f"half_life_days must be positive, got {half_life_days}")
    ref_ts = pd.Timestamp(ref_date)
    if isinstance(match_date, pd.Series):
        age_days = (ref_ts - pd.to_datetime(match_date)).dt.total_seconds() / 86_400.0
        age_days = age_days.clip(lower=0.0)
        return np.power(0.5, age_days / half_life_days)
    age_days = max(0.0, (ref_ts - pd.Timestamp(match_date)).total_seconds() / 86_400.0)
    return math.pow(0.5, age_days / half_life_days)


def combined_weight(
    matches: pd.DataFrame,
    *,
    ref_date: pd.Timestamp | datetime,
    half_life_days: float,
) -> pd.Series:
    """Return the per-row weight = time_decay * importance.

    Expects ``matches`` to have ``date`` (datetime) and ``tournament`` (string) columns.
    """
    if "date" not in matches.columns or "tournament" not in matches.columns:
        raise ValueError("matches must have 'date' and 'tournament' columns")
    decay = time_decay_weight(matches["date"], ref_date, half_life_days)
    importance = matches["tournament"].map(match_importance_weight).astype(float)
    return (decay * importance).rename("weight")
