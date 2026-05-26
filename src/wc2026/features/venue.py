"""Host-venue metadata + climate features for the WC 2026 corpus.

The static ``data/static/host_cities_climate.json`` table carries altitude
(metres) and June–July climate normals for every 2026 host venue. Two
helpers live here:

* :func:`venue_altitude_m` — synchronous lookup, no network.
* :func:`venue_wet_bulb_c` — first tries Open-Meteo's free forecast API
  for the given kickoff date (no API key required), then falls back to
  the climate-zone median in the static JSON when the API is unreachable.

Why this matters
----------------
Mexico City sits at 2 240 m — by far the highest 2026 venue — and the
Dallas/Miami/Monterrey corridor will routinely exceed 29 °C wet-bulb,
prompting FIFA's hydration-break rule. Both signals are absent from the
v1 PoissonDC and XGB feature pipelines; this module is the data layer
for adding them.

The DC core itself is left untouched (its analytic gradient doesn't
accept exogenous inputs without a rewrite). The XGB blend is the
intended consumer; see :mod:`wc2026.features.build_match_features` for
the integration point.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path("data/static/host_cities_climate.json")
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


@dataclass(frozen=True)
class VenueClimate:
    """Per-venue metadata derived from the static JSON."""

    city: str
    country: str
    lat: float
    lon: float
    altitude_m: int
    climate_zone: str
    typical_kickoff_temp_c: float
    typical_kickoff_wet_bulb_c: float


@lru_cache(maxsize=1)
def _load(path_str: str | None = None) -> dict[str, VenueClimate]:
    """Read the static JSON once per process. Tests can pass a fresh path."""
    p = Path(path_str) if path_str else DEFAULT_PATH
    raw = json.loads(p.read_text())
    return {
        c["city"]: VenueClimate(
            city=c["city"],
            country=c["country"],
            lat=float(c["lat"]),
            lon=float(c["lon"]),
            altitude_m=int(c["altitude_m"]),
            climate_zone=str(c["climate_zone"]),
            typical_kickoff_temp_c=float(c["typical_kickoff_temp_c"]),
            typical_kickoff_wet_bulb_c=float(c["typical_kickoff_wet_bulb_c"]),
        )
        for c in raw["cities"]
    }


def all_venues(path: Path | None = None) -> dict[str, VenueClimate]:
    """Return the in-memory venue dictionary keyed by city name."""
    return _load(str(path) if path is not None else None)


def venue_altitude_m(city: str, path: Path | None = None) -> int:
    """Altitude in metres for ``city``.

    Raises ``KeyError`` for unknown venues so the caller doesn't silently
    paper over a typo in the fixture metadata.
    """
    entry = all_venues(path).get(city)
    if entry is None:
        raise KeyError(f"unknown WC 2026 venue: {city!r}")
    return entry.altitude_m


def _wet_bulb_from_open_meteo(lat: float, lon: float, kickoff_date: date) -> float | None:
    """Single Open-Meteo call. Returns ``None`` on any failure.

    We ask for hourly ``wet_bulb_temperature_2m`` and pick the midday value
    (15:00 local) as a kickoff-time proxy. Open-Meteo's free tier doesn't
    require an API key.
    """
    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "start_date": kickoff_date.isoformat(),
        "end_date": kickoff_date.isoformat(),
        "hourly": "wet_bulb_temperature_2m",
        "timezone": "auto",
    }
    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        logger.debug("open-meteo wet-bulb fetch failed for (%s, %s)", lat, lon)
        return None
    times = data.get("hourly", {}).get("time", [])
    temps = data.get("hourly", {}).get("wet_bulb_temperature_2m", [])
    if not times or not temps or len(times) != len(temps):
        return None
    # Pick the 15:00 local-time slot; fall back to the daily mean.
    target = f"{kickoff_date.isoformat()}T15:00"
    for t, v in zip(times, temps, strict=True):
        if t == target and v is not None:
            return float(v)
    valid = [float(v) for v in temps if v is not None]
    return sum(valid) / len(valid) if valid else None


def venue_wet_bulb_c(
    city: str,
    kickoff_date: date,
    *,
    use_forecast: bool = True,
    path: Path | None = None,
) -> float:
    """Wet-bulb temperature (°C) at ``city`` on ``kickoff_date``.

    Tries Open-Meteo's free forecast first; falls back to the static
    climate-normal median when the API is unreachable, the date is
    outside the forecast window, or ``use_forecast`` is False (the
    default for tests).
    """
    entry = all_venues(path).get(city)
    if entry is None:
        raise KeyError(f"unknown WC 2026 venue: {city!r}")
    if use_forecast:
        live = _wet_bulb_from_open_meteo(entry.lat, entry.lon, kickoff_date)
        if live is not None:
            return live
    return entry.typical_kickoff_wet_bulb_c


__all__ = [
    "VenueClimate",
    "all_venues",
    "venue_altitude_m",
    "venue_wet_bulb_c",
]
