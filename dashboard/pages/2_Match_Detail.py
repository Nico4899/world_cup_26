"""Match Detail — per-match probability + 11x11 score heatmap + plain-language why."""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dashboard.components.api_client import (
    APIUnreachable,
    get_h2h,
    get_match,
    get_recent_form,
    render_unreachable_warning,
)

# Color tokens for W/D/L form bubbles.
_RESULT_COLOR = {"W": "#1f9d55", "D": "#888888", "L": "#d62728"}

DISPLAY_GOALS = 7  # truncate the 11×11 matrix for visual clarity

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

# Single API call — the /api/v1/matches/{id} endpoint returns top-5 scorelines
# AND the full 11×11 score_matrix, so the heatmap is the real distribution.
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

# Score heatmap from the full matrix returned by the API.
full_matrix = pred.get("score_matrix")
if full_matrix is None:
    st.warning("score_matrix missing from API response — heatmap unavailable.")
else:
    arr = np.asarray(full_matrix, dtype=float)
    # Crop to DISPLAY_GOALS; cells beyond are usually <0.1% combined.
    crop = arr[: DISPLAY_GOALS + 1, : DISPLAY_GOALS + 1]
    truncated_mass = float(arr.sum() - crop.sum())
    heatmap = px.imshow(
        crop,
        labels={"x": f"{fx['away_team']} goals", "y": f"{fx['home_team']} goals", "color": "P"},
        x=list(range(DISPLAY_GOALS + 1)),
        y=list(range(DISPLAY_GOALS + 1)),
        color_continuous_scale="Blues",
        aspect="equal",
        text_auto=".1%",
        zmin=0,
    )
    heatmap.update_layout(
        title=(
            f"Joint score probability (0–{DISPLAY_GOALS} shown; {truncated_mass:.1%} mass beyond)"
        ),
        height=460,
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

top1 = pred["top_scorelines"][0]
st.markdown(
    f"""
    - Expected goals: **{fx["home_team"]} {xg_h:.2f}** vs **{fx["away_team"]} {xg_a:.2f}**
    - {edge}, {home_adv_note}.
    - Top scoreline: **{top1["home_goals"]}–{top1["away_goals"]}** at {top1["probability"]:.1%}.
    - Remember: a 60% favourite still loses 40% of the time. These are probabilities, not predictions.
    """
)

st.divider()


def _render_form(team: str) -> None:
    """Render a team's last-5 form as coloured W/D/L bubbles + tooltip text."""
    st.markdown(f"**{team}** — last 5")
    try:
        form = get_recent_form(team, n=5)
    except APIUnreachable as exc:
        render_unreachable_warning(exc)
        return
    if not form:
        st.caption("_no recent matches in dataset_")
        return
    # Render each match as a colored badge in a row.
    badges_html = " ".join(
        f"<span style='background:{_RESULT_COLOR[m['result']]};color:white;"
        f"padding:3px 8px;border-radius:6px;margin-right:4px;font-weight:600' "
        f"title='{m['date']} {team} {m['goals_for']}-{m['goals_against']} {m['opponent']} "
        f"({m['venue']}, {m['tournament']})'>{m['result']}</span>"
        for m in form
    )
    st.markdown(badges_html, unsafe_allow_html=True)
    st.caption(
        " · ".join(
            f"{m['result']} {m['goals_for']}-{m['goals_against']} {'vs' if m['venue'] != 'away' else '@'} {m['opponent']}"
            for m in form
        )
    )


st.subheader("Recent form")
form_l, form_r = st.columns(2)
with form_l:
    _render_form(fx["home_team"])
with form_r:
    _render_form(fx["away_team"])

st.subheader("Head-to-head")
try:
    h2h = get_h2h(fx["home_team"], fx["away_team"], n=10)
except APIUnreachable as exc:
    render_unreachable_warning(exc)
else:
    if not h2h:
        st.caption(
            f"_{fx['home_team']} and {fx['away_team']} have never met in the dataset (1872 → present)._"
        )
    else:
        # Quick aggregate
        wins_home = sum(
            1
            for m in h2h
            if (m["home_team"] == fx["home_team"] and m["home_score"] > m["away_score"])
            or (m["away_team"] == fx["home_team"] and m["away_score"] > m["home_score"])
        )
        draws = sum(1 for m in h2h if m["home_score"] == m["away_score"])
        wins_away = len(h2h) - wins_home - draws
        st.caption(
            f"Last {len(h2h)} meetings — "
            f"**{fx['home_team']}** {wins_home} W · **draw** {draws} · **{fx['away_team']}** {wins_away} W"
        )
        rows = [
            {
                "Date": m["date"],
                "Home": m["home_team"],
                "Score": f"{m['home_score']}–{m['away_score']}",
                "Away": m["away_team"],
                "Tournament": m["tournament"],
                "Neutral": "✓" if m["neutral"] else "",
            }
            for m in h2h
        ]
        st.dataframe(rows, hide_index=True, width="stretch")
