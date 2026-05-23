"""Match Detail — per-match probability + 11x11 score heatmap + plain-language why."""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dashboard.components.api_client import (
    APIUnreachable,
    get_match,
    get_prediction,
    render_unreachable_warning,
)

st.title("Match detail")

# Read match_id from query params; fall back to a selectbox.
params = st.query_params
preselected_id = None
if "match_id" in params:
    try:
        preselected_id = int(params["match_id"])
    except ValueError:
        preselected_id = None

match_id = st.number_input(
    "Match ID (0..71)",
    min_value=0,
    max_value=71,
    value=preselected_id if preselected_id is not None else 0,
    step=1,
)

try:
    detail = get_match(int(match_id))
except APIUnreachable as exc:
    render_unreachable_warning(exc)
    st.stop()

fx = detail["fixture"]
pred = detail["prediction"]

st.subheader(f"{fx['home_team']} vs {fx['away_team']}")
st.caption(
    f"Group {fx['group']} · {fx['city']}, {fx['country']} · {fx['date']} · "
    + ("neutral venue" if fx["neutral"] else f"{fx['home_team']} at home")
)

col_a, col_b, col_c = st.columns(3)
col_a.metric(fx["home_team"], f"{pred['outcome']['home_win']:.1%}")
col_b.metric("Draw", f"{pred['outcome']['draw']:.1%}")
col_c.metric(fx["away_team"], f"{pred['outcome']['away_win']:.1%}")

# 1X2 bar chart at the top.
bar = go.Figure(
    go.Bar(
        x=[pred["outcome"]["home_win"], pred["outcome"]["draw"], pred["outcome"]["away_win"]],
        y=[fx["home_team"], "Draw", fx["away_team"]],
        orientation="h",
        marker_color=["#1f77b4", "#7f7f7f", "#d62728"],
        text=[
            f"{pred['outcome']['home_win']:.1%}",
            f"{pred['outcome']['draw']:.1%}",
            f"{pred['outcome']['away_win']:.1%}",
        ],
        textposition="outside",
    )
)
bar.update_layout(
    height=220,
    xaxis={"range": [0, 1.0], "tickformat": ".0%"},
    showlegend=False,
    margin={"l": 80, "r": 40, "t": 10, "b": 10},
)
st.plotly_chart(bar, config={"displayModeBar": False})

# Build a Plotly imshow heatmap of the score matrix (truncated 0..7 for display).
DISPLAY_GOALS = 7
# We need a fresh prediction call to recover the full top_scorelines; reuse the cached prediction.
try:
    full = get_prediction(fx["home_team"], fx["away_team"], neutral=fx["neutral"])
except APIUnreachable as exc:
    render_unreachable_warning(exc)
    st.stop()

# We only have the top 5 scorelines, not the full matrix from the API.
# Reconstruct the visible portion of the heatmap from the top scorelines we DO have
# (rest of the cells will read 0 — accurate enough for a visual cue and avoids needing
# a separate endpoint that returns the full matrix).
matrix = np.zeros((DISPLAY_GOALS + 1, DISPLAY_GOALS + 1))
for sc in full["top_scorelines"]:
    h = min(int(sc["home_goals"]), DISPLAY_GOALS)
    a = min(int(sc["away_goals"]), DISPLAY_GOALS)
    matrix[h, a] = sc["probability"]

heatmap = px.imshow(
    matrix,
    labels={"x": f"{fx['away_team']} goals", "y": f"{fx['home_team']} goals", "color": "P"},
    x=list(range(DISPLAY_GOALS + 1)),
    y=list(range(DISPLAY_GOALS + 1)),
    color_continuous_scale="Blues",
    aspect="equal",
    text_auto=".1%",
)
heatmap.update_layout(
    title="Most likely scorelines (top-5 highlighted; other cells suppressed)",
    height=420,
    margin={"l": 40, "r": 10, "t": 50, "b": 40},
)
st.plotly_chart(heatmap, config={"displayModeBar": False})

st.divider()
st.subheader("Why this prediction")

xg_h = pred["expected_home_goals"]
xg_a = pred["expected_away_goals"]
xg_diff = xg_h - xg_a
home_adv_note = (
    "and **at home** in their host country (home advantage applied)"
    if not fx["neutral"]
    else "**at a neutral venue** (home advantage suppressed)"
)

if abs(xg_diff) < 0.20:
    edge = "The model sees this as roughly even on expected goals"
elif xg_diff > 0:
    edge = f"The model gives **{fx['home_team']} a +{xg_diff:.2f} expected-goal edge**"
else:
    edge = f"The model gives **{fx['away_team']} a +{-xg_diff:.2f} expected-goal edge**"

st.markdown(
    f"""
    - Expected goals: **{fx["home_team"]} {xg_h:.2f}** vs **{fx["away_team"]} {xg_a:.2f}**
    - {edge}, {home_adv_note}.
    - Top scoreline: **{full["top_scorelines"][0]["home_goals"]}–{full["top_scorelines"][0]["away_goals"]}** at
      {full["top_scorelines"][0]["probability"]:.1%}.
    - Remember: a 60% favourite still loses 40% of the time. These are probabilities, not predictions.
    """
)
