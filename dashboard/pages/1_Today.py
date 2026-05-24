"""Today — prediction cards for the selected matchday."""

from __future__ import annotations

from datetime import date

import plotly.graph_objects as go
import streamlit as st
from dashboard.components.api_client import (
    APIUnreachable,
    get_match,
    get_standings,
    list_matches,
    render_unreachable_warning,
)
from dashboard.components.forecast_header import render_forecast_header
from dashboard.components.plot_config import PLOTLY_CONFIG
from dashboard.components.team_assets import render_team_chip, render_versus_header


def _navigate_to_detail(match_id: int) -> None:
    """on_click handler: stash match_id in URL params, then jump to Match Detail."""
    st.query_params["match_id"] = str(match_id)
    st.switch_page("pages/2_Match_Detail.py")


st.title("Today's predictions")
render_forecast_header()

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
            st.markdown(
                render_versus_header(m["home_team"], m["away_team"]),
                unsafe_allow_html=True,
            )
            kickoff_bits = []
            utc_kickoff = m.get("utc_kickoff")
            if utc_kickoff:
                # Strip seconds + take the "HH:MM" tail so the caption stays compact.
                kickoff_bits.append(f"⏱ {utc_kickoff[11:16]} UTC")
            kickoff_bits.append(f"Group {m['group']}")
            kickoff_bits.append(f"{m['city']}, {m['country']}")
            kickoff_bits.append("neutral" if m["neutral"] else "home advantage")
            st.caption(" · ".join(kickoff_bits))
            try:
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
                config=PLOTLY_CONFIG,
            )
            xg_h = pred["expected_home_goals"]
            xg_a = pred["expected_away_goals"]
            st.metric(label="Expected goals", value=f"{xg_h:.2f}  –  {xg_a:.2f}")
            st.caption("Most likely scorelines:")
            for sc in pred["top_scorelines"][:3]:
                st.write(f"- **{sc['home_goals']}–{sc['away_goals']}** ({sc['probability']:.1%})")
            st.button(
                "View match detail →",
                key=f"view-{m['match_id']}",
                on_click=_navigate_to_detail,
                args=(m["match_id"],),
            )

st.divider()
st.subheader("Group-stage advancement (across all 12 groups)")
st.caption(
    "Stacked-bar advancement probabilities from the Monte Carlo simulator. "
    "Each row is one group; segments are P(1st) · P(2nd) · P(3rd → R32) · P(eliminated). "
    "For per-team detail see the **Groups** page."
)

try:
    standings = get_standings(n_sims=2000, seed=42)
except APIUnreachable as exc:
    render_unreachable_warning(exc)
else:
    st.caption(f"Based on {standings['n_sims']} simulations.")
    # Render compact 4-column row per group; each cell is the group name plus a tiny
    # progress-bar string with the highest-prob advancing team highlighted.
    for row_start in range(0, len(standings["groups"]), 4):
        cols = st.columns(4)
        for col, block in zip(cols, standings["groups"][row_start : row_start + 4], strict=False):
            with col:
                st.markdown(f"**Group {block['group']}**")
                for team in block["teams"]:
                    advance_p = team["p_first"] + team["p_second"] + team["p_third_advance"]
                    bar_filled = round(advance_p * 12)
                    bar = "█" * bar_filled + "░" * (12 - bar_filled)
                    st.markdown(
                        f"<code style='font-size:0.85em'>{bar}</code> "
                        f"{render_team_chip(team['team'])} "
                        f"<small>· {advance_p:.0%} adv</small>",
                        unsafe_allow_html=True,
                    )
