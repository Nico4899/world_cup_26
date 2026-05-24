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
from dashboard.components.forecast_header import render_forecast_header
from dashboard.components.plot_config import PLOTLY_CONFIG
from dashboard.components.team_assets import (
    crest_img_html,
    get_team_assets,
)

_RESULT_COLOR = {"W": "#1f9d55", "D": "#888888", "L": "#d62728"}

st.title("Team Profile")
render_forecast_header()

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
    st.plotly_chart(bar, config=PLOTLY_CONFIG)
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
    st.plotly_chart(line, config=PLOTLY_CONFIG)
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

st.divider()

# --- FIFA ranking history ---------------------------------------------------

st.subheader("FIFA Men's Ranking history")
try:
    rankings = get_team_fifa_rankings(team)
except (APIUnreachable, httpx.HTTPStatusError):
    rankings = {"team": team, "history": []}

ranking_history = rankings.get("history") or []
if not ranking_history:
    st.caption(
        "_No FIFA ranking snapshots on file yet. The monthly `fifa_ranking_refresh` "
        "scheduler job populates `raw_fifa_rankings`._"
    )
else:
    latest = ranking_history[-1]
    rank_cols = st.columns(3)
    rank_cols[0].metric("Current rank", f"#{latest['rank']}")
    rank_cols[1].metric(
        "Points", f"{latest['points']:.0f}" if latest.get("points") is not None else "—"
    )
    delta = None
    if latest.get("previous_rank") is not None:
        # Lower numeric rank = better, so previous - current is the "up" delta.
        delta = int(latest["previous_rank"]) - int(latest["rank"])
    rank_cols[2].metric(
        "Δ since prev. snapshot",
        f"{delta:+d}" if delta is not None else "—",
        help="Positive = climbed, negative = fell.",
    )
    if len(ranking_history) > 1:
        line = go.Figure(
            go.Scatter(
                x=[p["ranking_date"] for p in ranking_history],
                y=[p["rank"] for p in ranking_history],
                mode="lines+markers",
                line={"color": "#9467bd"},
            )
        )
        # Lower rank = better, so invert the y axis for an intuitive "up = better" read.
        line.update_layout(
            height=240,
            xaxis_title="Snapshot date",
            yaxis_title="FIFA rank (lower = better)",
            yaxis={"autorange": "reversed"},
            margin={"l": 60, "r": 30, "t": 10, "b": 40},
        )
        st.plotly_chart(line, config=PLOTLY_CONFIG)

st.divider()

# --- xG form ---------------------------------------------------------------

st.subheader("xG form")
try:
    xg = get_team_xg_form(team)
except (APIUnreachable, httpx.HTTPStatusError):
    xg = {"team": team, "last_5": {"matches": 0}, "last_10": {"matches": 0}}

last5 = xg.get("last_5") or {"matches": 0}
last10 = xg.get("last_10") or {"matches": 0}
if last10.get("matches", 0) == 0:
    st.caption(
        "_No xG records for this team yet. The weekly `fbref_refresh` + manual "
        "`statsbomb_refresh` jobs populate `raw_match_xg`._"
    )
else:
    xg_cols = st.columns(2)
    for col, split, label in (
        (xg_cols[0], last5, "Last 5"),
        (xg_cols[1], last10, "Last 10"),
    ):
        with col:
            n = int(split.get("matches", 0) or 0)
            st.markdown(f"**{label}** — {n} match{'es' if n != 1 else ''}")
            if n == 0:
                st.caption("_no rows in window_")
                continue
            sub = st.columns(3)
            xf = split.get("xg_for")
            xa = split.get("xg_against")
            xd = split.get("xg_diff")
            sub[0].metric("xG for / match", f"{xf:.2f}" if xf is not None else "—")
            sub[1].metric("xG against / match", f"{xa:.2f}" if xa is not None else "—")
            sub[2].metric(
                "xG diff",
                f"{xd:+.2f}" if xd is not None else "—",
                help="Average per-match xG_for − xG_against.",
            )

st.divider()

# --- Squad roster ----------------------------------------------------------

st.subheader("Squad")
try:
    squad = get_team_squad(team)
except (APIUnreachable, httpx.HTTPStatusError):
    squad = {"team": team, "players": []}

players = squad.get("players") or []
if not players:
    st.caption(
        "_No squad snapshot on file. Run the manual `wikipedia_squads_refresh` "
        "job from the Operator page once squads are announced._"
    )
else:
    snap = squad.get("snapshot_date")
    src = squad.get("tournament") or "tournament"
    st.caption(
        f"{len(players)} players · snapshot {snap or 'unknown date'} ({src})."
    )
    rows = [
        {
            "#": p.get("shirt_number") if p.get("shirt_number") is not None else "—",
            "Player": p["player_name"],
            "Pos": p.get("position") or "—",
            "Club": p.get("club") or "—",
            "Born": p.get("birth_date") or "—",
            "Caps": p.get("caps") if p.get("caps") is not None else "—",
            "Goals": p.get("goals") if p.get("goals") is not None else "—",
        }
        for p in players
    ]
    st.dataframe(rows, hide_index=True, width="stretch")
