"""Tests for the host-venue altitude + wet-bulb helpers."""

from __future__ import annotations

from datetime import date

import pytest

from wc2026.features import venue


def test_all_venues_returns_16_entries() -> None:
    cities = venue.all_venues()
    assert len(cities) == 16
    assert {c.country for c in cities.values()} == {
        "United States",
        "Mexico",
        "Canada",
    }


def test_venue_altitude_mexico_city_is_2240_m() -> None:
    # Estadio Azteca sits at roughly 2 240 m — the only 2026 venue above 2 000 m.
    assert venue.venue_altitude_m("Mexico City") == 2240


def test_venue_altitude_sea_level_venues_under_50_m() -> None:
    for city in ("Miami", "New York/New Jersey", "Houston", "Vancouver", "Seattle"):
        assert venue.venue_altitude_m(city) < 50


def test_venue_altitude_unknown_city_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        venue.venue_altitude_m("Atlantis")


def test_wet_bulb_falls_back_to_climate_median_when_forecast_disabled() -> None:
    # use_forecast=False bypasses the Open-Meteo call entirely, so this is a
    # deterministic lookup against the static climate-normals table.
    miami = venue.venue_wet_bulb_c("Miami", date(2026, 6, 15), use_forecast=False)
    vancouver = venue.venue_wet_bulb_c("Vancouver", date(2026, 6, 15), use_forecast=False)
    assert miami > 25.0  # Miami: hot + humid, wet-bulb ~27 °C
    assert vancouver < 17.0  # Vancouver: maritime cool


def test_wet_bulb_fallback_when_open_meteo_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Network failure path: helper returns the climate median, not None."""
    monkeypatch.setattr(venue, "_wet_bulb_from_open_meteo", lambda *_args, **_kw: None)
    miami_static = venue.venue_wet_bulb_c("Miami", date(2026, 6, 15), use_forecast=False)
    miami_live = venue.venue_wet_bulb_c("Miami", date(2026, 6, 15), use_forecast=True)
    assert miami_live == miami_static


def test_wet_bulb_uses_live_value_when_open_meteo_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: returned value is the API value, not the climate median."""
    monkeypatch.setattr(venue, "_wet_bulb_from_open_meteo", lambda *_args, **_kw: 28.5)
    miami_live = venue.venue_wet_bulb_c("Miami", date(2026, 6, 15), use_forecast=True)
    assert miami_live == pytest.approx(28.5)
