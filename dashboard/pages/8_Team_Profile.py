"""Team Profile — Elo history + recent-5 + WC 2026 advancement probabilities.

Picks the team from a dropdown driven by the WC 2026 fixture list, then pulls
its Elo timeline, recent form, and per-round championship probability from
the latest persisted Monte Carlo run.
"""

from __future__ import annotations

import httpx
import plotly.graph_objects as go
import streamlit as st
from dashboard.components.api_client import (
    APIUnreachable,
    get_json,
    get_recent_form,
    get_team_elo_history,
    get_team_fifa_rankings,
    get_team_squad,
    get_team_tournament_probs,
    get_team_xg_form,
    render_unreachable_warning,
)
from dashboard.components.team_assets import (
    crest_img_html,
    get_team_assets,
)

_RESULT_COLOR = {"W": "#1f9d55", "D": "#888888", "L": "#d62728"}

st.title("Team Profile")

# Pull the 48 WC 2026 teams from /api/v1/matches and dedupe.
try:
    fixtures_raw = get_json("/api/v1/matches")
except APIUnreachable as exc:
    render_unreachable_warning(exc)
    st.stop()

teams = sorted({m["home_team"] for m in fixtures_raw} | {m["away_team"] for m in fixtures_raw})
team = st.selectbox("Team", teams, index=teams.index("Argentina") if "Argentina" in teams else 0)

# --- Header chip with crest --------------------------------------------------

assets = get_team_assets(team)
st.markdown(
    f"<div style='display:flex;align-items:center;gap:12px;font-size:1.4rem'>"
    f"{crest_img_html(assets, height_px=40)}<strong>{team}</strong></div>",
    unsafe_allow_html=True,
)

# --- WC 2026 advancement probabilities --------------------------------------

st.subheader("WC 2026 advancement probabilities")
try:
    probs = get_team_tournament_probs(team)
except (APIUnreachable, httpx.HTTPStatusError):
    probs = {"team": team, "run_id": None}

if probs.get("run_id") is None:
    st.caption(
        "_No Monte Carlo run on disk yet. Once the daily refit + persist job "
        "(`scripts/rerun_monte_carlo.py`) has written a run, these probabilities "
        "appear here._"
    )
elif probs.get("champion_p") is None:
    st.caption(
        f"_Latest run (id {probs['run_id']}, {probs.get('n_sims', '?')} sims) "
        "had no row for this team — perhaps the team list changed since the "
        "last refit._"
    )
else:
    cols = st.columns(4)
    cols[0].metric("Champion", f"{(probs.get('champion_p') or 0):.1%}")
    cols[1].metric("Final", f"{(probs.get('final_p') or 0):.1%}")
    cols[2].metric("Semifinal", f"{(probs.get('semifinal_p') or 0):.1%}")
    cols[3].metric("Quarterfinal", f"{(probs.get('quarterfinal_p') or 0):.1%}")

    # Path-to-final bar chart: cumulative prob of reaching each round.
    stages = [
        ("Group winner", probs.get("group_winner_p")),
        ("Group runner-up", probs.get("group_runner_up_p")),
        ("Advance to R32", probs.get("advance_r32_p")),
        ("Reach R16", probs.get("advance_r16_p")),
        ("Reach QF", probs.get("quarterfinal_p")),
        ("Reach SF", probs.get("semifinal_p")),
        ("Reach Final", probs.get("final_p")),
        ("Champion", probs.get("champion_p")),
    ]
    labels = [s[0] for s in stages]
    values = [(v or 0.0) for _, v in stages]
    bar = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color="#1f77b4",
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
        )
    )
    bar.update_layout(
        height=320,
        xaxis={"range": [0, 1.0], "tickformat": ".0%"},
        margin={"l": 120, "r": 60, "t": 10, "b": 30},
        yaxis={"autorange": "reversed"},
    )
    st.plotly_chart(bar, config={"displayModeBar": False})
    st.caption(
        f"From persisted run {probs['run_id']} ({probs.get('n_sims', '?')} sims, "
        f"model {probs.get('model_version', '?')}). Probabilities update on the "
        "next conditional rerun after a match finishes."
    )

st.divider()

# --- Elo history -----------------------------------------------------------

st.subheader("Elo rating history")
try:
    elo = get_team_elo_history(team)
except (APIUnreachable, httpx.HTTPStatusError):
    elo = {"team": team, "history": []}

history = elo.get("history") or []
if not history:
    st.caption(
        "_No Elo snapshots on disk for this team. The daily `elo_refresh` "
        "scheduler job populates `raw_elo_snapshots`._"
    )
else:
    line = go.Figure(
        go.Scatter(
            x=[p["snapshot_date"] for p in history],
            y=[p["rating"] for p in history],
            mode="lines+markers",
            line={"color": "#1f77b4"},
        )
    )
    line.update_layout(
        height=300,
        xaxis_title="Snapshot date",
        yaxis_title="Elo rating",
        margin={"l": 60, "r": 30, "t": 10, "b": 40},
    )
    st.plotly_chart(line, config={"displayModeBar": False})
    st.caption(
        f"{len(history)} daily snapshots. Most recent: {history[-1]['snapshot_date']} → "
        f"{history[-1]['rating']:.1f}"
    )

st.divider()

# --- Recent form -----------------------------------------------------------

st.subheader("Recent form (last 10 internationals)")
try:
    form = get_recent_form(team, n=10)
except APIUnreachable as exc:
    render_unreachable_warning(exc)
    form = []

if not form:
    st.caption("_No recent matches in the dataset._")
else:
    badges = " ".join(
        f"<span style='background:{_RESULT_COLOR[m['result']]};color:white;"
        f"padding:3px 8px;border-radius:6px;margin-right:4px;font-weight:600' "
        f"title='{m['date']} {team} {m['goals_for']}-{m['goals_against']} {m['opponent']} "
        f"({m['venue']}, {m['tournament']})'>{m['result']}</span>"
        for m in form
    )
    st.markdown(badges, unsafe_allow_html=True)
    st.caption(
        " · ".join(
            f"{m['result']} {m['goals_for']}-{m['goals_against']} "
            f"{'vs' if m['venue'] != 'away' else '@'} {m['opponent']}"
            for m in form
        )
    )
