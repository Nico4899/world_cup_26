"""Match Detail — per-match probability + 11x11 score heatmap + plain-language why.

Phase 6 addition: a "LIVE" section appears whenever the API reports a non-
pre-match win-prob source for the fixture. It renders the current state, an
in-running win-prob bar, and a per-event timeline chart. The page auto-
refreshes every 5 seconds while the match is live (via a meta-refresh tag —
no external dependency).
"""

from __future__ import annotations

import httpx
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dashboard.components.api_client import (
    APIUnreachable,
    get_explanation,
    get_h2h,
    get_live_history,
    get_match,
    get_prediction,
    get_recent_form,
    get_team_elo_history,
    render_unreachable_warning,
)
from dashboard.components.plot_config import PLOTLY_CONFIG
from dashboard.components.team_assets import render_team_chip, render_versus_header


def _latest_elo(team: str) -> float | None:
    """Pull the most-recent Elo rating for ``team`` from the API.

    Returns ``None`` when no snapshot exists or the API is unreachable so
    the narrative can degrade gracefully to an xG-only sentence.
    """
    try:
        payload = get_team_elo_history(team)
    except (APIUnreachable, httpx.HTTPStatusError):
        return None
    history = (payload or {}).get("history") or []
    if not history:
        return None
    try:
        return float(history[-1]["rating"])
    except (KeyError, TypeError, ValueError):
        return None

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

st.markdown(
    render_versus_header(fx["home_team"], fx["away_team"]),
    unsafe_allow_html=True,
)
_when = fx["date"]
if fx.get("utc_kickoff"):
    # ISO 8601 "YYYY-MM-DDTHH:MM:SS..." — slice the HH:MM and join.
    _when = f"{fx['date']} {fx['utc_kickoff'][11:16]} UTC"
st.caption(
    f"Group {fx['group']} · {fx['city']}, {fx['country']} · {_when} · "
    + ("neutral venue" if fx["neutral"] else f"{fx['home_team']} at home")
)

def _render_outcome_popover(
    *, column, label: str, probability: float, class_name: str
) -> None:
    """Per-outcome clickable cell that reveals the SHAP top-3 driver list.

    Trades ``st.metric``'s bold visual for a popover button so every probability
    on the page is reachable from a single click — the spec's "interpretability
    as headline feature" principle.
    """
    with column, st.popover(
        f"{label}\n\n**{probability:.1%}**", width="stretch"
    ):
        st.markdown(f"**Why {label} = {probability:.1%}?**")
        try:
            ex = get_explanation(int(match_id), class_name=class_name, top_n=3)
        except (APIUnreachable, httpx.HTTPStatusError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 503:
                st.caption(
                    "_SHAP explanations require the optional XGB classifier. "
                    "Train it with `uv run python scripts/refit_xgb.py` and restart_."
                )
            else:
                st.caption("_explanation endpoint unavailable_")
            return
        contribs = (ex or {}).get("contributions") or []
        if not contribs:
            st.caption("_no contributions returned_")
            return
        for c in contribs:
            sign = "↑" if c["contribution"] >= 0 else "↓"
            st.write(
                f"- {sign} **{c['feature']}** "
                f"(value {('—' if c.get('value') is None else f'{c['value']:+.3f}')}, "
                f"contribution {c['contribution']:+.3f})"
            )


col_a, col_b, col_c = st.columns(3)
_render_outcome_popover(
    column=col_a,
    label=fx["home_team"],
    probability=pred["outcome"]["home_win"],
    class_name="home_win",
)
_render_outcome_popover(
    column=col_b, label="Draw", probability=pred["outcome"]["draw"], class_name="draw"
)
_render_outcome_popover(
    column=col_c,
    label=fx["away_team"],
    probability=pred["outcome"]["away_win"],
    class_name="away_win",
)

# --- Phase 5 blend overlay --------------------------------------------------
# Opt-in sidebar toggle that surfaces the Poisson × XGB geometric mean
# alongside the headline Poisson numbers. Pulled via the /predictions
# endpoint with ?blend=true so the Match Detail score-heatmap (which is
# Poisson-only by construction) stays untouched.
with st.sidebar:
    st.markdown("### XGB blend (Phase 5)")
    show_blend = st.checkbox(
        "Show Poisson × XGB blend",
        value=False,
        help=(
            "Overlays the Phase 5 XGBoost classifier alongside the Poisson outcome. "
            "The blend is opt-in: WC 2018/2022 hindcasts show it regressing log-loss, "
            "so it's surfaced as a research artefact rather than the default."
        ),
    )
    blend_weight = st.slider(
        "Poisson weight",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        disabled=not show_blend,
        help="Geometric-mean mixing weight; XGB gets 1 − weight.",
    )

if show_blend:
    try:
        blended = get_prediction(
            fx["home_team"],
            fx["away_team"],
            neutral=fx["neutral"],
            blend=True,
            blend_weight=blend_weight,
        )
    except APIUnreachable:
        blended = None  # warning already rendered above
    except httpx.HTTPStatusError as exc:
        st.caption(f"_blend overlay unavailable: HTTP {exc.response.status_code}_")
        blended = None
    payload = (blended or {}).get("blend")
    if blended is not None and payload is None:
        st.caption(
            "_XGB artefact not loaded server-side — train it with "
            "`uv run python scripts/refit_xgb.py` and restart the API._"
        )
    elif payload is not None:
        st.markdown("**Blended outcome (Poisson × XGB)**")
        b_a, b_b, b_c = st.columns(3)
        b_a.metric(fx["home_team"], f"{payload['blended']['home_win']:.1%}")
        b_b.metric("Draw", f"{payload['blended']['draw']:.1%}")
        b_c.metric(fx["away_team"], f"{payload['blended']['away_win']:.1%}")
        st.caption(
            f"Poisson weight {payload['weight']:.2f}, XGB weight {1 - payload['weight']:.2f}. "
            f"Poisson H/D/A: {payload['poisson']['home_win']:.1%} / "
            f"{payload['poisson']['draw']:.1%} / {payload['poisson']['away_win']:.1%} · "
            f"XGB H/D/A: {payload['xgb']['home_win']:.1%} / "
            f"{payload['xgb']['draw']:.1%} / {payload['xgb']['away_win']:.1%}."
        )

# --- Phase 6 live section --------------------------------------------------
# A non-"poisson_pre_match" snapshot source means events have been ingested
# for this fixture; render the in-running state above the historical detail.
live_payload = None
try:
    live_payload = get_live_history(int(match_id))
except APIUnreachable:
    # The /live endpoints share the same upstream as the rest of the page —
    # the earlier render_unreachable_warning would have stopped us already, so
    # we shouldn't get here. Belt-and-braces fallback: silently skip.
    live_payload = None
except httpx.HTTPStatusError as exc:
    # An older API (pre-Phase 6) returns 404 for /live/{id}/history. Treat
    # that as "no live data" so the page still renders against legacy
    # deployments — the rest of the Match Detail view doesn't depend on it.
    if exc.response.status_code != 404:
        raise
    live_payload = None

snapshot = (live_payload or {}).get("snapshot")
events = (live_payload or {}).get("events") or []
if snapshot is not None and snapshot.get("win_prob_source") in (
    "live_win_prob",
    "final",
):
    is_live = snapshot["win_prob_source"] == "live_win_prob"
    if is_live:
        # Embed a meta-refresh so the page reloads every 5 seconds while the
        # match is in progress. The browser handles it; no streamlit-autorefresh
        # dependency needed.
        st.markdown("<meta http-equiv='refresh' content='5'>", unsafe_allow_html=True)

    st.divider()
    badge = "🔴 LIVE" if is_live else "✅ FULL TIME"
    st.markdown(
        f"<span style='background:#d62728;color:white;padding:4px 10px;"
        f"border-radius:6px;font-weight:700'>{badge}</span>",
        unsafe_allow_html=True,
    )

    score_text = f"{snapshot['home_score']}–{snapshot['away_score']}"
    state_text = f"min {snapshot['minute']} · last: {snapshot['last_event_type']}"
    if snapshot["home_red_cards"] or snapshot["away_red_cards"]:
        state_text += f" · 🟥 {snapshot['home_red_cards']}–{snapshot['away_red_cards']}"
    st.subheader(f"{fx['home_team']} {score_text} {fx['away_team']}")
    st.caption(state_text)

    live_col_a, live_col_b, live_col_c = st.columns(3)
    wp = snapshot["win_prob"]
    live_col_a.metric(f"{fx['home_team']} (live)", f"{wp['home_win']:.1%}")
    live_col_b.metric("Draw (live)", f"{wp['draw']:.1%}")
    live_col_c.metric(f"{fx['away_team']} (live)", f"{wp['away_win']:.1%}")

    if events:
        # Stack the three probability series over the per-event timeline.
        minutes = [ev["minute"] for ev in events]
        line = go.Figure()
        line.add_trace(
            go.Scatter(
                x=minutes,
                y=[ev["win_prob"]["home_win"] for ev in events],
                name=fx["home_team"],
                mode="lines+markers",
                line={"color": "#1f77b4"},
            )
        )
        line.add_trace(
            go.Scatter(
                x=minutes,
                y=[ev["win_prob"]["draw"] for ev in events],
                name="Draw",
                mode="lines+markers",
                line={"color": "#7f7f7f"},
            )
        )
        line.add_trace(
            go.Scatter(
                x=minutes,
                y=[ev["win_prob"]["away_win"] for ev in events],
                name=fx["away_team"],
                mode="lines+markers",
                line={"color": "#d62728"},
            )
        )
        # Annotation pins for goals + red cards
        for ev in events:
            if ev["event_type"] == "GOAL":
                line.add_vline(
                    x=ev["minute"],
                    line_color="#1f9d55",
                    line_dash="dash",
                    annotation_text=f"⚽ {ev['team'] or ''}".strip(),
                    annotation_position="top",
                )
            elif ev["event_type"] == "FT_WHISTLE":
                line.add_vline(
                    x=ev["minute"],
                    line_color="#888888",
                    line_dash="dot",
                    annotation_text="FT",
                    annotation_position="top",
                )
        line.update_layout(
            title="Live win probability",
            xaxis_title="minute",
            yaxis={"range": [0, 1.0], "tickformat": ".0%"},
            height=300,
            margin={"l": 60, "r": 20, "t": 50, "b": 40},
        )
        st.plotly_chart(line, config=PLOTLY_CONFIG)

    st.divider()

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
st.plotly_chart(bar, config=PLOTLY_CONFIG)

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
    st.plotly_chart(heatmap, config=PLOTLY_CONFIG)

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

# Sentence-level Elo anchor: when both teams have an Elo snapshot, surface
# the gap explicitly. Spec example: "Brazil's Elo is 121 above Tunisia's;
# that translates to a +0.74 expected-goal edge on neutral ground."
elo_home = _latest_elo(fx["home_team"])
elo_away = _latest_elo(fx["away_team"])
elo_sentence: str | None = None
if elo_home is not None and elo_away is not None:
    elo_diff = elo_home - elo_away
    if abs(elo_diff) < 5:
        elo_sentence = (
            f"Elo is essentially level ({elo_home:.0f} vs {elo_away:.0f}); "
            f"the model leans on the bivariate Poisson + home-advantage term for the edge."
        )
    elif elo_diff > 0:
        elo_sentence = (
            f"**{fx['home_team']}'s Elo is +{elo_diff:.0f} above {fx['away_team']}'s** "
            f"({elo_home:.0f} vs {elo_away:.0f}); that translates to a "
            f"**{xg_diff:+.2f} expected-goal edge** here."
        )
    else:
        elo_sentence = (
            f"**{fx['away_team']}'s Elo is +{-elo_diff:.0f} above {fx['home_team']}'s** "
            f"({elo_away:.0f} vs {elo_home:.0f}); that translates to a "
            f"**{xg_diff:+.2f} expected-goal edge** here."
        )

top1 = pred["top_scorelines"][0]
bullets = [
    f"- Expected goals: **{fx['home_team']} {xg_h:.2f}** vs **{fx['away_team']} {xg_a:.2f}**",
    f"- {edge}, {home_adv_note}.",
]
if elo_sentence:
    bullets.append(f"- {elo_sentence}")
bullets.extend(
    [
        f"- Top scoreline: **{top1['home_goals']}–{top1['away_goals']}** at {top1['probability']:.1%}.",
        "- Remember: a 60% favourite still loses 40% of the time. These are probabilities, not predictions.",
    ]
)
st.markdown("\n".join(bullets))

st.divider()

# --- Phase 9 SHAP panel ----------------------------------------------------

st.subheader("Why (model contributions)")
class_label = st.radio(
    "Outcome to explain",
    options=("home_win", "draw", "away_win"),
    horizontal=True,
    format_func=lambda v: {
        "home_win": fx["home_team"],
        "draw": "Draw",
        "away_win": fx["away_team"],
    }[v],
)
explanation = None
try:
    explanation = get_explanation(int(match_id), class_name=class_label, top_n=6)
except APIUnreachable:
    pass  # already warned about above
except httpx.HTTPStatusError as exc:
    if exc.response.status_code == 503:
        st.caption(
            "_SHAP explanations require the optional XGB classifier. Train it with_ "
            "`uv run python scripts/refit_xgb.py` _and restart the API to enable this panel._"
        )
    elif exc.response.status_code == 404:
        st.caption("_`/api/v1/explain` not deployed yet — restart the API after Phase 5._")
    else:
        raise

if explanation is not None:
    contribs = explanation.get("contributions") or []
    if not contribs:
        st.caption("_No contributions returned._")
    else:
        # Order by absolute contribution descending (the API already does this);
        # render as a horizontal bar chart with signed values + colour coding.
        names = [c["feature"] for c in contribs]
        values = [c["contribution"] for c in contribs]
        colors = ["#1f9d55" if v > 0 else "#d62728" for v in values]
        shap_fig = go.Figure(
            go.Bar(
                x=values,
                y=names,
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.3f}" for v in values],
                textposition="outside",
            )
        )
        shap_fig.update_layout(
            height=320,
            xaxis_title=f"SHAP value (signed; positive ⇒ pushes {class_label} prob up)",
            yaxis={"autorange": "reversed"},
            margin={"l": 140, "r": 60, "t": 10, "b": 40},
        )
        st.plotly_chart(shap_fig, config=PLOTLY_CONFIG)
        # Per-row "feature value" annotations as a compact caption.
        rows = [
            {
                "Feature": c["feature"],
                "Value": "n/a" if c.get("value") is None else f"{c['value']:+.3f}",
                "Contribution": f"{c['contribution']:+.4f}",
            }
            for c in contribs
        ]
        st.dataframe(rows, hide_index=True, width="stretch")

st.divider()


def _render_form(team: str) -> None:
    """Render a team's last-5 form as coloured W/D/L bubbles + tooltip text."""
    st.markdown(f"{render_team_chip(team, bold=True)} — last 5", unsafe_allow_html=True)
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
