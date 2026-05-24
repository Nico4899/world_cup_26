"""Bracket Realisation — one (or several) sampled knockout outcomes from the Monte Carlo.

Phase 9 enhancement: the page now supports a multi-seed scenario explorer.
Pick the number of scenarios and the dashboard draws that many independent
realisations side-by-side, so you can see how stable a final-four or champion
prediction is across different random tournaments.

This is **not** a click-to-set interactive bracket simulator — the plan
flagged that as a custom-JS effort that the Plotly fallback can't fully
replicate. The scenario-comparison view captures most of the same value
(seeing variance across realisations) without the JS toolchain.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from dashboard.components.api_client import (
    APIUnreachable,
    get_json,
    render_unreachable_warning,
)
from dashboard.components.forecast_header import render_forecast_header
from dashboard.components.team_assets import render_team_chip

st.title("Knockout bracket realisation(s)")
render_forecast_header()

st.caption(
    "Each seed gives one Monte Carlo sample of the full knockout bracket. "
    "Compare scenarios to see how much the predicted champion / finalists "
    "vary across realisations. Per-team **probabilities** (the aggregate over "
    "10 000+ sims) are on the Groups page."
)

mode = st.radio(
    "View mode",
    options=("Single seed", "Scenario comparison"),
    horizontal=True,
)


def _fetch_bracket(seed_value: int) -> dict | None:
    try:
        return get_json("/api/v1/tournament/bracket", params={"seed": seed_value})
    except APIUnreachable as exc:
        render_unreachable_warning(exc)
        return None


def _render_bracket_detail(data: dict) -> None:
    st.markdown(
        f"🏆 **Champion (seed {data['seed']})**: "
        f"{render_team_chip(data['champion'], bold=True)}",
        unsafe_allow_html=True,
    )
    df = pd.DataFrame(data["matches"])
    df["score"] = df["regulation_score"].apply(lambda s: f"{s[0]}–{s[1]}")
    df["decided"] = df["decided_in"].map(
        {"regulation": "90'", "extra_time": "AET", "shootout": "pens"}
    )
    df = df[["round", "match_id", "home_team", "score", "away_team", "winner", "decided"]]
    df.columns = ["Round", "#", "Home", "Score", "Away", "Winner", "Decided"]
    round_order = ["R32", "R16", "QF", "SF", "Final"]
    for r in round_order:
        sub = df[df["Round"] == r]
        if sub.empty:
            continue
        with st.expander(f"{r} — {len(sub)} match(es)", expanded=(r in ("SF", "Final"))):
            st.dataframe(sub.drop(columns=["Round"]), hide_index=True, width="stretch")


if mode == "Single seed":
    seed = st.number_input("Seed", min_value=0, max_value=1_000_000, value=42, step=1)
    data = _fetch_bracket(int(seed))
    if data is None:
        st.stop()
    _render_bracket_detail(data)
else:
    st.caption(
        "Draws several independent brackets with consecutive seeds, then shows "
        "how often each team reaches the SF / final / lifts the trophy across "
        "this small sample."
    )
    n_scenarios = st.slider("Number of scenarios", min_value=2, max_value=8, value=4, step=1)
    base_seed = st.number_input(
        "Base seed (first scenario)", min_value=0, max_value=1_000_000, value=42, step=1
    )
    scenarios = []
    for i in range(int(n_scenarios)):
        data = _fetch_bracket(int(base_seed) + i)
        if data is None:
            st.stop()
        scenarios.append(data)

    # Aggregate: how often each team reaches the final / wins across scenarios.
    finals_count: dict[str, int] = {}
    champs_count: dict[str, int] = {}
    sf_count: dict[str, int] = {}
    for s in scenarios:
        finalists = next((m for m in s["matches"] if m["round"] == "Final"), None)
        sfs = [m for m in s["matches"] if m["round"] == "SF"]
        for sf in sfs:
            sf_count[sf["home_team"]] = sf_count.get(sf["home_team"], 0) + 1
            sf_count[sf["away_team"]] = sf_count.get(sf["away_team"], 0) + 1
        if finalists is not None:
            finals_count[finalists["home_team"]] = finals_count.get(finalists["home_team"], 0) + 1
            finals_count[finalists["away_team"]] = finals_count.get(finalists["away_team"], 0) + 1
        champs_count[s["champion"]] = champs_count.get(s["champion"], 0) + 1

    n = len(scenarios)
    rows = sorted(
        (
            {
                "Team": team,
                "SFs": sf_count.get(team, 0),
                "Finals": finals_count.get(team, 0),
                "Champion": champs_count.get(team, 0),
                "Champion %": f"{champs_count.get(team, 0) / n:.0%}",
            }
            for team in {*sf_count, *finals_count, *champs_count}
        ),
        key=lambda r: r["Champion"] * 100 + r["Finals"] * 10 + r["SFs"],
        reverse=True,
    )
    st.subheader(f"Across {n} scenarios")
    st.dataframe(rows, hide_index=True, width="stretch")

    cols = st.columns(min(n, 4))
    for i, s in enumerate(scenarios):
        col = cols[i % len(cols)]
        with col:
            st.caption(f"Seed {s['seed']}")
            st.markdown(
                render_team_chip(s["champion"], bold=True),
                unsafe_allow_html=True,
            )

    # Drill-down: pick one of the scenarios to inspect in full.
    pick = st.selectbox(
        "Show full bracket for",
        [f"Seed {s['seed']} — champion: {s['champion']}" for s in scenarios],
        index=0,
    )
    chosen_seed = int(pick.split()[1].rstrip("—").strip())
    chosen = next(s for s in scenarios if int(s["seed"]) == chosen_seed)
    _render_bracket_detail(chosen)
