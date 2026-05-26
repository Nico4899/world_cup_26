"""Unit tests for the match-feature orchestrator."""

from __future__ import annotations

import math
from datetime import date

import pandas as pd

from wc2026.features.build_match_features import (
    FeatureSources,
    MatchSpec,
    build_features_for_match,
    build_features_for_matches,
)
from wc2026.features.venue import VenueClimate
from wc2026.models.poisson_dc import PoissonDC


def _historical_matches() -> pd.DataFrame:
    """A small training set so PoissonDC can fit Argentina and France."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-09-04",
                    "2025-10-12",
                    "2025-11-15",
                    "2026-03-22",
                    "2026-05-04",
                    "2026-05-30",
                    "2026-06-03",
                ]
            ),
            "home_team": [
                "Argentina",
                "France",
                "Argentina",
                "France",
                "Argentina",
                "France",
                "Argentina",
            ],
            "away_team": [
                "Brazil",
                "Spain",
                "Uruguay",
                "Italy",
                "Chile",
                "Belgium",
                "Mexico",
            ],
            "home_score": [3, 2, 1, 2, 2, 4, 3],
            "away_score": [1, 0, 1, 1, 1, 1, 0],
            "neutral": [False, False, False, False, False, False, False],
        }
    )


def _xg_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_date": pd.to_datetime(
                ["2026-03-22", "2026-05-04", "2026-06-03", "2025-11-15", "2026-05-30"]
            ),
            "team": ["Argentina", "Argentina", "Argentina", "France", "France"],
            "xg_for": [2.4, 2.1, 2.8, 1.5, 1.7],
            "xg_against": [0.8, 0.9, 0.5, 1.1, 1.3],
        }
    )


def _fitted_poisson() -> PoissonDC:
    model = PoissonDC()
    return model.fit(_historical_matches())


def test_build_features_returns_full_schema() -> None:
    spec = MatchSpec(
        match_date=date(2026, 6, 11),
        home_team="Argentina",
        away_team="France",
        neutral=True,
    )
    sources = FeatureSources()  # no upstream data
    row = build_features_for_match(spec, sources)
    expected_keys = {
        "match_date",
        "home_team",
        "away_team",
        "elo_diff",
        "fifa_rank_diff",
        "xg_form_diff",
        "rest_days_diff",
        "squad_age_diff",
        "is_neutral",
        "is_host_home",
        "is_host_away",
        "poisson_exp_home_goals",
        "poisson_exp_away_goals",
        "poisson_p_home",
        "poisson_p_draw",
        "poisson_p_away",
        "source_snapshots",
        "venue_altitude_m",
        "venue_wet_bulb_c",
    }
    assert set(row.keys()) == expected_keys


def test_build_features_emits_none_for_missing_sources() -> None:
    spec = MatchSpec(date(2026, 6, 11), "Argentina", "France")
    row = build_features_for_match(spec, FeatureSources())
    assert row["elo_diff"] is None
    assert row["fifa_rank_diff"] is None
    assert row["xg_form_diff"] is None
    assert row["rest_days_diff"] is None
    assert row["squad_age_diff"] is None
    assert row["poisson_p_home"] is None


def test_venue_features_emit_none_when_no_climate_source() -> None:
    """venue_altitude_m + venue_wet_bulb_c default to None when sources lack
    a climate lookup OR the spec has no venue_city — the row schema stays
    stable for the XGB consumer."""
    spec = MatchSpec(date(2026, 6, 11), "Argentina", "France", venue_city=None)
    row = build_features_for_match(spec, FeatureSources())
    assert row["venue_altitude_m"] is None
    assert row["venue_wet_bulb_c"] is None


def test_venue_features_emitted_when_climate_source_present() -> None:
    azteca = VenueClimate(
        city="Mexico City",
        country="Mexico",
        lat=19.3,
        lon=-99.15,
        altitude_m=2240,
        climate_zone="subtropical_highland",
        typical_kickoff_temp_c=22.0,
        typical_kickoff_wet_bulb_c=14.0,
    )
    sources = FeatureSources(venue_climate={"Mexico City": azteca})
    spec = MatchSpec(
        date(2026, 6, 11),
        "Mexico",
        "Saudi Arabia",
        venue_city="Mexico City",
    )
    row = build_features_for_match(spec, sources)
    assert row["venue_altitude_m"] == 2240.0
    assert row["venue_wet_bulb_c"] == 14.0


def test_venue_wet_bulb_override_takes_precedence() -> None:
    """A per-match (city, date) override (e.g. live Open-Meteo forecast)
    replaces the static climate-median fallback."""
    miami = VenueClimate(
        city="Miami",
        country="United States",
        lat=25.96,
        lon=-80.24,
        altitude_m=2,
        climate_zone="tropical_monsoon",
        typical_kickoff_temp_c=30.0,
        typical_kickoff_wet_bulb_c=27.0,
    )
    match_date = date(2026, 6, 15)
    sources = FeatureSources(
        venue_climate={"Miami": miami},
        venue_wet_bulb_override={("Miami", match_date): 29.3},
    )
    spec = MatchSpec(match_date, "Argentina", "Mexico", venue_city="Miami")
    row = build_features_for_match(spec, sources)
    assert row["venue_wet_bulb_c"] == 29.3


def test_neutral_and_host_flags_set_correctly() -> None:
    spec = MatchSpec(date(2026, 6, 11), "Mexico", "Senegal", neutral=False)
    row = build_features_for_match(spec, FeatureSources())
    assert row["is_neutral"] == 0
    assert row["is_host_home"] == 1
    assert row["is_host_away"] == 0
    spec2 = MatchSpec(date(2026, 6, 11), "Argentina", "United States", neutral=True)
    row2 = build_features_for_match(spec2, FeatureSources())
    assert row2["is_neutral"] == 1
    assert row2["is_host_home"] == 0
    assert row2["is_host_away"] == 1


def test_elo_diff_signed_home_minus_away() -> None:
    spec = MatchSpec(date(2026, 6, 11), "Argentina", "France")
    sources = FeatureSources(elo_by_team={"Argentina": 2150.0, "France": 2030.0})
    row = build_features_for_match(spec, sources)
    assert row["elo_diff"] == 120.0


def test_fifa_rank_diff_negative_when_home_ranked_better() -> None:
    """Lower rank = better; home rank 1 vs away rank 6 → diff = -5."""
    spec = MatchSpec(date(2026, 6, 11), "Argentina", "France")
    sources = FeatureSources(fifa_rank_by_team={"Argentina": 1, "France": 6})
    row = build_features_for_match(spec, sources)
    assert row["fifa_rank_diff"] == -5


def test_xg_form_diff_uses_provided_history() -> None:
    spec = MatchSpec(date(2026, 6, 11), "Argentina", "France")
    sources = FeatureSources(xg_history=_xg_history(), xg_form_window=5)
    row = build_features_for_match(spec, sources)
    # Argentina recent xG: (2.4+2.1+2.8)/3 = 2.43, conceded 0.73 → diff +1.70
    # France recent xG: (1.5+1.7)/2 = 1.6, conceded 1.2 → diff +0.4
    # xg_form_diff = 1.70 - 0.40 = 1.30
    assert row["xg_form_diff"] is not None
    assert math.isclose(row["xg_form_diff"], 1.30, abs_tol=0.01)


def test_rest_days_diff_uses_provided_history() -> None:
    spec = MatchSpec(date(2026, 6, 11), "Argentina", "France")
    sources = FeatureSources(matches=_historical_matches())
    row = build_features_for_match(spec, sources)
    # Argentina last match 2026-06-03 (8 days); France last match 2026-05-30 (12 days).
    assert row["rest_days_diff"] == 8 - 12


def test_squad_age_diff_uses_provided_means() -> None:
    spec = MatchSpec(date(2026, 6, 11), "Argentina", "France")
    sources = FeatureSources(squad_age_by_team={"Argentina": 27.8, "France": 26.4})
    row = build_features_for_match(spec, sources)
    assert math.isclose(row["squad_age_diff"], 1.4, abs_tol=1e-9)


def test_poisson_features_populated_for_fitted_model() -> None:
    spec = MatchSpec(date(2026, 6, 11), "Argentina", "France", neutral=True)
    sources = FeatureSources(poisson_model=_fitted_poisson())
    row = build_features_for_match(spec, sources)
    assert row["poisson_exp_home_goals"] > 0
    assert row["poisson_exp_away_goals"] > 0
    total = row["poisson_p_home"] + row["poisson_p_draw"] + row["poisson_p_away"]
    assert abs(total - 1.0) < 1e-9


def test_poisson_features_none_when_team_not_in_fitted_set() -> None:
    spec = MatchSpec(date(2026, 6, 11), "Argentina", "Atlantis")
    sources = FeatureSources(poisson_model=_fitted_poisson())
    row = build_features_for_match(spec, sources)
    assert row["poisson_exp_home_goals"] is None
    assert row["poisson_p_home"] is None


def test_source_snapshots_round_trips_into_row() -> None:
    spec = MatchSpec(date(2026, 6, 11), "Mexico", "Senegal")
    sources = FeatureSources(
        snapshot_meta={
            "elo_snapshot_date": "2026-05-23",
            "model_version": "poisson_dc.v1",
        }
    )
    row = build_features_for_match(spec, sources)
    assert row["source_snapshots"] == {
        "elo_snapshot_date": "2026-05-23",
        "model_version": "poisson_dc.v1",
    }


def test_build_features_for_matches_returns_one_row_per_spec() -> None:
    specs = [
        MatchSpec(date(2026, 6, 11), "Argentina", "France"),
        MatchSpec(date(2026, 6, 12), "Brazil", "Germany"),
        MatchSpec(date(2026, 6, 13), "Mexico", "USA"),
    ]
    df = build_features_for_matches(specs, FeatureSources())
    assert len(df) == 3
    assert list(df["home_team"]) == ["Argentina", "Brazil", "Mexico"]
