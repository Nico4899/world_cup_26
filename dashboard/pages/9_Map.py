"""Map — 16 WC 2026 host cities, click-to-filter by venue.

Plots every host city on a PyDeck ScatterplotLayer (richer than `st.map`:
per-pin tooltip + per-country fill colour) and surfaces a city dropdown
that filters the WC 2026 fixtures list to matches at that venue. The
``HOST_CITIES`` dict below holds the curated lat/lng tuples — no external
geocoding API needed.
"""

from __future__ import annotations

import pandas as pd
import pydeck as pdk
import streamlit as st
from dashboard.components.api_client import APIUnreachable, get_json, render_unreachable_warning

_COUNTRY_FILL: dict[str, list[int]] = {
    # Roughly matches the host countries' flag accent so each cluster is
    # recognisable at a glance. RGBA channel triples, alpha implied 200.
    "United States": [31, 119, 180],  # navy
    "Mexico": [44, 160, 44],  # green
    "Canada": [214, 39, 40],  # red
}

# Curated host-city coordinates (publicly documented FIFA venues for WC 2026).
HOST_CITIES: dict[str, dict] = {
    # United States (11 venues)
    "Atlanta": {"lat": 33.7553, "lon": -84.4006, "country": "United States"},
    "Boston": {"lat": 42.0909, "lon": -71.2643, "country": "United States"},
    "Dallas": {"lat": 32.7473, "lon": -97.0945, "country": "United States"},
    "Houston": {"lat": 29.6847, "lon": -95.4107, "country": "United States"},
    "Kansas City": {"lat": 39.0489, "lon": -94.4839, "country": "United States"},
    "Los Angeles": {"lat": 33.9535, "lon": -118.3392, "country": "United States"},
    "Miami": {"lat": 25.9580, "lon": -80.2389, "country": "United States"},
    "New York/New Jersey": {"lat": 40.8135, "lon": -74.0745, "country": "United States"},
    "Philadelphia": {"lat": 39.9008, "lon": -75.1675, "country": "United States"},
    "San Francisco Bay Area": {"lat": 37.4030, "lon": -121.9700, "country": "United States"},
    "Seattle": {"lat": 47.5952, "lon": -122.3316, "country": "United States"},
    # Mexico (3 venues)
    "Mexico City": {"lat": 19.3029, "lon": -99.1505, "country": "Mexico"},
    "Guadalajara": {"lat": 20.6818, "lon": -103.4626, "country": "Mexico"},
    "Monterrey": {"lat": 25.6691, "lon": -100.2453, "country": "Mexico"},
    # Canada (2 venues)
    "Toronto": {"lat": 43.6332, "lon": -79.4196, "country": "Canada"},
    "Vancouver": {"lat": 49.2767, "lon": -123.1119, "country": "Canada"},
}

st.title("Host-city map")

st.caption(
    "16 host venues across USA (11) + Mexico (3) + Canada (2). "
    "Pick a city below to filter the fixture list to matches at that venue."
)

# --- Map -------------------------------------------------------------------

map_df = pd.DataFrame(
    [
        {
            "city": city,
            "lat": meta["lat"],
            "lon": meta["lon"],
            "country": meta["country"],
            "fill": _COUNTRY_FILL.get(meta["country"], [127, 127, 127]),
        }
        for city, meta in HOST_CITIES.items()
    ]
)
# PyDeck ScatterplotLayer: per-pin tooltip (city + country) + country fill.
# Radius is in metres at the equator; 30km reads well at zoom 3.
layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_df,
    get_position="[lon, lat]",
    get_fill_color="fill",
    get_radius=30_000,
    pickable=True,
    opacity=0.85,
    stroked=True,
    get_line_color=[255, 255, 255],
    line_width_min_pixels=1,
)
deck = pdk.Deck(
    layers=[layer],
    initial_view_state=pdk.ViewState(latitude=35.0, longitude=-100.0, zoom=2.6, pitch=0),
    tooltip={"text": "{city} ({country})"},
    map_style=None,
)
st.pydeck_chart(deck, width="stretch")

# --- City filter + fixture list -------------------------------------------

try:
    fixtures = get_json("/api/v1/matches")
except APIUnreachable as exc:
    render_unreachable_warning(exc)
    st.stop()

# The Jürisoo dataset stores city under each fixture; intersect with our list.
fixture_cities = sorted({m["city"] for m in fixtures})
# Some fixture city strings won't match our curated dict 1:1 (e.g. "East
# Rutherford" vs our "New York/New Jersey"). Surface the union so the user
# can still filter by the dataset's literal city name.
all_cities = sorted(set(HOST_CITIES.keys()) | set(fixture_cities))
choice = st.selectbox("Filter fixtures by host city", ["(all)", *all_cities])

if choice != "(all)":
    filtered = [m for m in fixtures if m["city"] == choice]
    if not filtered:
        st.info(
            f"No fixtures recorded at {choice!r}. (The Jürisoo dataset may use a "
            "slightly different city label — try the dropdown alternatives.)"
        )
else:
    filtered = fixtures

st.subheader("All 72 fixtures" if choice == "(all)" else f"{len(filtered)} fixture(s) at {choice}")
rows = [
    {
        "Match #": m["match_id"],
        "Date": m["date"],
        "Group": m["group"],
        "Home": m["home_team"],
        "Away": m["away_team"],
        "City": m["city"],
        "Country": m["country"],
    }
    for m in filtered
]
st.dataframe(rows, hide_index=True, width="stretch")
