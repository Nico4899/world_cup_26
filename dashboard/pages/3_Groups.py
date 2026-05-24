"""Groups — per-team advancement probabilities for each of the 12 groups."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from dashboard.components.api_client import (
    APIUnreachable,
    get_json,
    render_unreachable_warning,
)
from dashboard.components.forecast_header import render_forecast_header
from dashboard.components.plot_config import PLOTLY_CONFIG
from dashboard.components.team_assets import render_team_chip

st.title("Group-stage advancement probabilities")
render_forecast_header()

st.caption(
    "Each row shows where the model thinks a team will finish. Top 2 + 8 best 3rd-placed "
    "teams advance to the Round of 32. Bars stack: 1st (dark blue), 2nd (sky), "
    "3rd→R32 (amber), 3rd-out (light), 4th (grey)."
)

n_sims = st.sidebar.slider(
    "Monte Carlo simulations", min_value=200, max_value=10_000, value=2000, step=200
)

try:
    data = get_json("/api/v1/tournament/standings", params={"n_sims": n_sims, "seed": 42})
except APIUnreachable as exc:
    render_unreachable_warning(exc)
    st.stop()

# Phase 8 provenance: surface whether these probabilities come from a persisted
# Monte Carlo run (post-result conditional rerun) or from a cold in-process
# simulation. When persisted, also show the run id + model version so an operator
# can correlate against the scheduler log.
source = data.get("source")
run_id = data.get("run_id")
model_version = data.get("model_version")
if source == "persisted" and run_id is not None:
    st.caption(
        f"Based on {data['n_sims']} simulations · persisted run **#{run_id}** "
        f"(model `{model_version or 'unknown'}`). Updates after each completed match."
    )
else:
    st.caption(
        f"Based on {data['n_sims']} simulations · in-process run "
        "(no persisted Monte Carlo run on file yet — run `scripts/rerun_monte_carlo.py`)."
    )


def _group_fig(block: dict) -> go.Figure:
    teams = [r["team"] for r in block["teams"]]
    p_first = [r["p_first"] for r in block["teams"]]
    p_second = [r["p_second"] for r in block["teams"]]
    p_third_adv = [r["p_third_advance"] for r in block["teams"]]
    # Old persisted runs (pre-Phase-12) lack the granular split — fall back to a
    # 50/50 distribution of the eliminated mass so the bar still totals to 1.
    p_third_out = [r.get("p_third_out", r["p_eliminated"] / 2) for r in block["teams"]]
    p_fourth = [r.get("p_fourth", r["p_eliminated"] / 2) for r in block["teams"]]

    fig = go.Figure()
    for x, name, color in (
        (p_first, "1st", "#1f4e79"),
        (p_second, "2nd", "#5b9bd5"),
        (p_third_adv, "3rd → R32", "#ed7d31"),
        (p_third_out, "3rd-out", "#d9a679"),
        (p_fourth, "4th", "#a6a6a6"),
    ):
        fig.add_trace(
            go.Bar(
                y=teams,
                x=x,
                orientation="h",
                name=name,
                marker_color=color,
                text=[f"{p:.0%}" if p > 0.05 else "" for p in x],
                textposition="inside",
            )
        )
    fig.update_layout(
        barmode="stack",
        height=180,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        xaxis={"range": [0, 1.001], "showticklabels": False, "showgrid": False},
        yaxis={"autorange": "reversed"},
        showlegend=False,
        title=f"Group {block['group']}",
    )
    return fig


# 3 columns x 4 rows = 12 groups
cols_per_row = 3
group_list = data["groups"]
for row_start in range(0, len(group_list), cols_per_row):
    row = group_list[row_start : row_start + cols_per_row]
    cols = st.columns(cols_per_row)
    for col, block in zip(cols, row, strict=False):
        with col:
            st.plotly_chart(_group_fig(block), config=PLOTLY_CONFIG)

st.divider()
st.subheader("Headline: top 10 championship probabilities")
header_cols = st.columns([3, 1, 1, 1, 1])
for hdr, name in zip(header_cols, ("Team", "Champion", "Final", "Semi", "Quarter"), strict=True):
    hdr.markdown(f"**{name}**")
for h in data["headline"]:
    row_cols = st.columns([3, 1, 1, 1, 1])
    row_cols[0].markdown(render_team_chip(h["team"], bold=True), unsafe_allow_html=True)
    row_cols[1].markdown(f"{h['p_champion']:.1%}")
    row_cols[2].markdown(f"{h['p_final']:.1%}")
    row_cols[3].markdown(f"{h['p_sf']:.1%}")
    row_cols[4].markdown(f"{h['p_qf']:.1%}")
