"""Build the materialised feature row consumed by Phase 5's XGBoost classifier.

One row per (match_date, home_team, away_team). The orchestrator pulls from
every Phase 2/3 source we have on disk and merges them into the schema of
``db.models.MatchFeatures``. Sources are independently optional — a missing
source yields ``NaN`` for the features it would have produced, and the
materialiser still emits the row so downstream consumers can rely on a
consistent shape.

The output of this module is intentionally **dict-of-numerics** so it can be
fed directly to SQLAlchemy ``insert`` or to ``pd.DataFrame`` for ad-hoc model
experimentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from wc2026.features.host_team import is_host
from wc2026.features.rest_days import rest_days_diff
from wc2026.features.travel import travel_km_diff as compute_travel_km_diff
from wc2026.features.venue import VenueClimate
from wc2026.features.xg_form import compute_form_features, xg_form_diff
from wc2026.models.poisson_dc import PoissonDC


@dataclass(frozen=True)
class MatchSpec:
    """One match for which we want a feature row.

    ``neutral`` mirrors the convention in PoissonDC.expected_goals — True
    when neither team is the host of record (e.g. World Cup matches at a
    non-host venue).

    ``venue_city`` is optional; when set + the matching venue is in
    :class:`FeatureSources.venue_climate`, the orchestrator emits
    ``venue_altitude_m`` and ``venue_wet_bulb_c`` columns.
    """

    match_date: date
    home_team: str
    away_team: str
    neutral: bool = False
    venue_city: str | None = None


@dataclass(frozen=True)
class FeatureSources:
    """Bundle of inputs for the feature orchestrator.

    Every field is optional. Callers should populate whichever sources they
    have on disk; the orchestrator emits NaN for the features whose upstream
    source is missing.

    * ``elo_by_team``: ``{team_name: rating}`` (from latest eloratings snapshot)
    * ``fifa_rank_by_team``: ``{team_name: rank}`` (lower = better)
    * ``xg_history``: per-(match, team) rows with columns
      ``match_date, team, xg_for, xg_against`` — fed to ``compute_form_features``
    * ``squad_age_by_team``: ``{team_name: mean_age_years}``
    * ``matches``: historical ``date, home_team, away_team`` rows for rest-days
    * ``poisson_model``: fitted PoissonDC; if None we skip the Poisson features
    * ``snapshot_meta``: free-form dict copied into ``source_snapshots``
    * ``xg_form_window``: how many past matches the rolling-xG mean averages
    """

    elo_by_team: dict[str, float] | None = None
    fifa_rank_by_team: dict[str, int] | None = None
    xg_history: pd.DataFrame | None = None
    squad_age_by_team: dict[str, float] | None = None
    matches: pd.DataFrame | None = None
    poisson_model: PoissonDC | None = None
    snapshot_meta: dict[str, Any] = field(default_factory=dict)
    xg_form_window: int = 5
    # Optional host-venue climate lookup. When provided + the spec carries
    # ``venue_city``, the orchestrator emits ``venue_altitude_m`` +
    # ``venue_wet_bulb_c`` columns. Wet-bulb values are typically the
    # ``typical_kickoff_wet_bulb_c`` static normal; a per-match override
    # (e.g. live Open-Meteo forecast) is keyed by (city, match_date).
    venue_climate: dict[str, VenueClimate] | None = None
    venue_wet_bulb_override: dict[tuple[str, date], float] | None = None

    # Optional match-history DataFrame carrying venue coordinates per
    # row (columns: ``date, home_team, away_team, home_lat, home_lon``).
    # When provided + the spec carries ``venue_city``, the orchestrator
    # emits ``travel_km_diff``. Independent of ``matches`` because the
    # rest-days feature only needs (date, teams) but travel needs
    # (lat, lon) for each historical match too.
    travel_history: pd.DataFrame | None = None

    # Optional ``{team_name: squad_market_value_eur}`` lookup, sourced
    # from the manual ``transfermarkt_refresh`` job's parquet snapshot.
    # When provided, the orchestrator emits ``log_market_value_diff``.
    # ``None`` for either side ⇒ feature is ``None``.
    market_value_by_team: dict[str, float] | None = None


def _diff(a: float | int | None, b: float | int | None) -> float | None:
    """``a - b``, or ``None`` if either side is ``None``/``NaN``."""
    if a is None or b is None:
        return None
    try:
        if pd.isna(a) or pd.isna(b):
            return None
    except (TypeError, ValueError):
        return None
    return float(a) - float(b)


def _elo_diff(spec: MatchSpec, sources: FeatureSources) -> float | None:
    if not sources.elo_by_team:
        return None
    h = sources.elo_by_team.get(spec.home_team)
    a = sources.elo_by_team.get(spec.away_team)
    return _diff(h, a)


def _fifa_rank_diff(spec: MatchSpec, sources: FeatureSources) -> float | None:
    """``home_rank - away_rank``.

    Lower FIFA rank is better, so a *negative* diff means home is better-ranked.
    """
    if not sources.fifa_rank_by_team:
        return None
    h = sources.fifa_rank_by_team.get(spec.home_team)
    a = sources.fifa_rank_by_team.get(spec.away_team)
    return _diff(h, a)


def _xg_form_diff(spec: MatchSpec, sources: FeatureSources) -> float | None:
    if sources.xg_history is None or sources.xg_history.empty:
        return None
    form = compute_form_features(
        sources.xg_history,
        teams=[spec.home_team, spec.away_team],
        as_of=pd.Timestamp(spec.match_date),
        window=sources.xg_form_window,
    )
    diff = xg_form_diff(form, home=spec.home_team, away=spec.away_team)
    return None if pd.isna(diff) else float(diff)


def _rest_days_diff(spec: MatchSpec, sources: FeatureSources) -> float | None:
    if sources.matches is None or sources.matches.empty:
        return None
    diff = rest_days_diff(
        sources.matches,
        home=spec.home_team,
        away=spec.away_team,
        as_of=spec.match_date,
    )
    return None if diff is None else float(diff)


def _squad_age_diff(spec: MatchSpec, sources: FeatureSources) -> float | None:
    if not sources.squad_age_by_team:
        return None
    h = sources.squad_age_by_team.get(spec.home_team)
    a = sources.squad_age_by_team.get(spec.away_team)
    return _diff(h, a)


def _log_market_value_diff(spec: MatchSpec, sources: FeatureSources) -> float | None:
    """``log(home_market_value_eur) - log(away_market_value_eur)``.

    Returns ``None`` when the lookup is missing OR either side has no
    entry OR either value is ``<= 0`` (defensive against bad data).
    """
    if not sources.market_value_by_team:
        return None
    h = sources.market_value_by_team.get(spec.home_team)
    a = sources.market_value_by_team.get(spec.away_team)
    if h is None or a is None:
        return None
    try:
        h_f = float(h)
        a_f = float(a)
    except (TypeError, ValueError):
        return None
    if h_f <= 0.0 or a_f <= 0.0:
        return None
    import math  # noqa: PLC0415

    return math.log(h_f) - math.log(a_f)


def _travel_km_diff(spec: MatchSpec, sources: FeatureSources) -> float | None:
    """home minus away great-circle km from each team's previous venue.

    Requires both:
      - ``sources.venue_climate[spec.venue_city]`` for the current (lat, lon)
      - ``sources.travel_history`` with venue coords on prior matches
    Returns ``None`` when either input is missing.
    """
    if (
        spec.venue_city is None
        or sources.venue_climate is None
        or sources.travel_history is None
        or sources.travel_history.empty
    ):
        return None
    current = sources.venue_climate.get(spec.venue_city)
    if current is None:
        return None
    diff = compute_travel_km_diff(
        sources.travel_history,
        home=spec.home_team,
        away=spec.away_team,
        as_of=spec.match_date,
        current_lat=current.lat,
        current_lon=current.lon,
    )
    return None if diff is None else float(diff)


def _venue_features(spec: MatchSpec, sources: FeatureSources) -> dict[str, float | None]:
    """Altitude + wet-bulb for ``spec.venue_city``.

    Returns ``{None, None}`` when either the spec lacks a venue or the
    sources lack a climate lookup — keeping every row schema-stable.
    """
    if spec.venue_city is None or sources.venue_climate is None:
        return {"venue_altitude_m": None, "venue_wet_bulb_c": None}
    entry = sources.venue_climate.get(spec.venue_city)
    if entry is None:
        return {"venue_altitude_m": None, "venue_wet_bulb_c": None}
    if sources.venue_wet_bulb_override is not None:
        override = sources.venue_wet_bulb_override.get((spec.venue_city, spec.match_date))
        wet_bulb: float | None = (
            float(override) if override is not None else entry.typical_kickoff_wet_bulb_c
        )
    else:
        wet_bulb = entry.typical_kickoff_wet_bulb_c
    return {
        "venue_altitude_m": float(entry.altitude_m),
        "venue_wet_bulb_c": wet_bulb,
    }


def _poisson_features(spec: MatchSpec, sources: FeatureSources) -> dict[str, float | None]:
    if sources.poisson_model is None:
        return {
            "poisson_exp_home_goals": None,
            "poisson_exp_away_goals": None,
            "poisson_p_home": None,
            "poisson_p_draw": None,
            "poisson_p_away": None,
        }
    try:
        lam_h, lam_a = sources.poisson_model.expected_goals(
            spec.home_team, spec.away_team, neutral=spec.neutral
        )
        outcomes = sources.poisson_model.outcome_probs(
            spec.home_team, spec.away_team, neutral=spec.neutral
        )
    except (KeyError, ValueError):
        # Team not in the fitted set, or coefficients missing.
        return {
            "poisson_exp_home_goals": None,
            "poisson_exp_away_goals": None,
            "poisson_p_home": None,
            "poisson_p_draw": None,
            "poisson_p_away": None,
        }
    return {
        "poisson_exp_home_goals": float(lam_h),
        "poisson_exp_away_goals": float(lam_a),
        "poisson_p_home": float(outcomes["home_win"]),
        "poisson_p_draw": float(outcomes["draw"]),
        "poisson_p_away": float(outcomes["away_win"]),
    }


def build_features_for_match(spec: MatchSpec, sources: FeatureSources) -> dict[str, Any]:
    """Return a single feature dict matching the ``MatchFeatures`` schema.

    Numeric features whose upstream source is missing come back as ``None``;
    Phase 5's XGBoost imputes them, and the materialiser persists them as
    SQL NULL.
    """
    row: dict[str, Any] = {
        "match_date": spec.match_date,
        "home_team": spec.home_team,
        "away_team": spec.away_team,
        "elo_diff": _elo_diff(spec, sources),
        "fifa_rank_diff": _fifa_rank_diff(spec, sources),
        "xg_form_diff": _xg_form_diff(spec, sources),
        "rest_days_diff": _rest_days_diff(spec, sources),
        "squad_age_diff": _squad_age_diff(spec, sources),
        "is_neutral": int(spec.neutral),
        "is_host_home": is_host(spec.home_team),
        "is_host_away": is_host(spec.away_team),
        "source_snapshots": dict(sources.snapshot_meta) or None,
    }
    row.update(_poisson_features(spec, sources))
    row.update(_venue_features(spec, sources))
    row["travel_km_diff"] = _travel_km_diff(spec, sources)
    row["log_market_value_diff"] = _log_market_value_diff(spec, sources)
    return row


def build_features_for_matches(specs: list[MatchSpec], sources: FeatureSources) -> pd.DataFrame:
    """Vectorised wrapper: apply ``build_features_for_match`` to every spec."""
    return pd.DataFrame([build_features_for_match(s, sources) for s in specs])


__all__ = [
    "FeatureSources",
    "MatchSpec",
    "build_features_for_match",
    "build_features_for_matches",
]
