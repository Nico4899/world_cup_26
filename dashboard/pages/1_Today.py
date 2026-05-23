"""Today — prediction cards for the selected matchday."""

from __future__ import annotations

from datetime import date

import plotly.graph_objects as go
import streamlit as st
from dashboard.components.api_client import (
    APIUnreachable,
    list_matches,
    render_unreachable_warning,
)

st.title("Today's predictions")

WC_START = date(2026, 6, 11)
WC_END = date(2026, 6, 27)
default_date = WC_START

picked_date = st.date_input(
    "Matchday",
    value=default_date,
    min_value=WC_START,
    max_value=WC_END,
    help="The group-stage runs June 11 → 27, 2026. Knockout dates are pre-listed but the schedule is provisional.",
)

try:
    matches = list_matches(date=picked_date.isoformat())
except APIUnreachable as exc:
    render_unreachable_warning(exc)
    st.stop()

if not matches:
    st.info(f"No matches scheduled on {picked_date.isoformat()}.")
    st.stop()

st.caption(f"{len(matches)} match(es) on {picked_date.isoformat()}")


def _prob_bar(home: str, away: str, p_h: float, p_d: float, p_a: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=["1X2"],
            x=[p_h],
            orientation="h",
            name=home,
            marker_color="#1f77b4",
            text=f"{p_h:.0%}",
            textposition="inside",
        )
    )
    fig.add_trace(
        go.Bar(
            y=["1X2"],
            x=[p_d],
            orientation="h",
            name="Draw",
            marker_color="#7f7f7f",
            text=f"{p_d:.0%}",
            textposition="inside",
        )
    )
    fig.add_trace(
        go.Bar(
            y=["1X2"],
            x=[p_a],
            orientation="h",
            name=away,
            marker_color="#d62728",
            text=f"{p_a:.0%}",
            textposition="inside",
        )
    )
    fig.update_layout(
        barmode="stack",
        height=110,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        xaxis={"range": [0, 1], "showticklabels": False, "showgrid": False},
        yaxis={"showticklabels": False},
        showlegend=True,
        legend={"orientation": "h", "y": -0.4},
    )
    return fig


# Render each match in a row, two columns wide.
for i in range(0, len(matches), 2):
    row = matches[i : i + 2]
    cols = st.columns(2)
    for col, m in zip(cols, row, strict=False):
        with col, st.container(border=True):
            st.subheader(f"{m['home_team']} vs {m['away_team']}")
            st.caption(
                f"Group {m['group']} · {m['city']}, {m['country']} · "
                + ("neutral" if m["neutral"] else "home advantage")
            )
            try:
                from dashboard.components.api_client import get_match  # local import to share cache

                detail = get_match(m["match_id"])
            except APIUnreachable as exc:
                render_unreachable_warning(exc)
                continue
            pred = detail["prediction"]
            st.plotly_chart(
                _prob_bar(
                    m["home_team"],
                    m["away_team"],
                    pred["outcome"]["home_win"],
                    pred["outcome"]["draw"],
                    pred["outcome"]["away_win"],
                ),
                config={"displayModeBar": False},
            )
            xg_h = pred["expected_home_goals"]
            xg_a = pred["expected_away_goals"]
            st.metric(label="Expected goals", value=f"{xg_h:.2f}  –  {xg_a:.2f}")
            st.caption("Most likely scorelines:")
            for sc in pred["top_scorelines"]:
                st.write(f"- **{sc['home_goals']}–{sc['away_goals']}** ({sc['probability']:.1%})")
            st.page_link(
                f"pages/2_Match_Detail.py?match_id={m['match_id']}",
                label="View match detail →",
            )
