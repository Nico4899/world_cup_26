"""Groups — per-team advancement probabilities for each of the 12 groups."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from dashboard.components.api_client import (
    APIUnreachable,
    get_json,
    render_unreachable_warning,
)

st.title("Group-stage advancement probabilities")

st.caption(
    "Each row shows where the model thinks a team will finish. Top 2 + 8 best 3rd-placed "
    "teams advance to the Round of 32. Bars stack: 1st (blue), 2nd (sky), 3rd-advance "
    "(amber), eliminated (grey)."
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
    p_third = [r["p_third_advance"] for r in block["teams"]]
    p_out = [r["p_eliminated"] for r in block["teams"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=teams,
            x=p_first,
            orientation="h",
            name="1st",
            marker_color="#1f4e79",
            text=[f"{p:.0%}" if p > 0.05 else "" for p in p_first],
            textposition="inside",
        )
    )
    fig.add_trace(
        go.Bar(
            y=teams,
            x=p_second,
            orientation="h",
            name="2nd",
            marker_color="#5b9bd5",
            text=[f"{p:.0%}" if p > 0.05 else "" for p in p_second],
            textposition="inside",
        )
    )
    fig.add_trace(
        go.Bar(
            y=teams,
            x=p_third,
            orientation="h",
            name="3rd → R32",
            marker_color="#ed7d31",
            text=[f"{p:.0%}" if p > 0.05 else "" for p in p_third],
            textposition="inside",
        )
    )
    fig.add_trace(
        go.Bar(
            y=teams,
            x=p_out,
            orientation="h",
            name="eliminated",
            marker_color="#a6a6a6",
            text=[f"{p:.0%}" if p > 0.05 else "" for p in p_out],
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
            st.plotly_chart(_group_fig(block), config={"displayModeBar": False})

st.divider()
st.subheader("Headline: top 10 championship probabilities")
st.dataframe(
    [
        {
            "Team": h["team"],
            "Champion": f"{h['p_champion']:.1%}",
            "Final": f"{h['p_final']:.1%}",
            "Semi": f"{h['p_sf']:.1%}",
            "Quarter": f"{h['p_qf']:.1%}",
        }
        for h in data["headline"]
    ],
    hide_index=True,
    width="stretch",
)
