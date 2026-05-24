"""Track Record — reliability diagrams + headline metrics on past tournaments.

Runs the WC 2022 / WC 2018 hindcasts inline (cached for the session) and plots
per-outcome reliability. This is the calibration honesty page: the model's
trustworthiness lives or dies here.

Phase 7 addition: a live WC 2026 panel at the top, fed by
``/api/v1/track-record/wc2026``. Stays empty (with a "no completed matches
yet" caption) until the live event poller starts writing FT_WHISTLE rows.
"""

from __future__ import annotations

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dashboard.components.api_client import APIUnreachable, get_json
from dashboard.components.plot_config import PLOTLY_CONFIG

st.title("Track record")

st.caption(
    "Calibration check on completed World Cups (out-of-sample). Each point is one "
    "predicted-probability bin; the dashed line is perfect calibration."
)

# --- WC 2026 live rolling calibration --------------------------------------

st.subheader("WC 2026 — live rolling calibration")
try:
    wc2026_payload = get_json("/api/v1/track-record/wc2026")
except APIUnreachable:
    wc2026_payload = None
    st.info(
        "API unreachable. Start it with `uv run uvicorn wc2026.api.main:app` to see "
        "live WC 2026 calibration here."
    )
except httpx.HTTPStatusError as exc:
    wc2026_payload = None
    if exc.response.status_code == 404:
        st.caption(
            "The running API doesn't yet expose `/track-record/wc2026` — restart it "
            "after deploying Phase 7."
        )
    else:
        raise

if wc2026_payload is not None:
    n_completed = int(wc2026_payload.get("n_completed", 0))
    if n_completed == 0:
        st.caption(
            "_No completed WC 2026 matches with both a pre-match prediction and an "
            "FT-whistle event on record yet. Once the live event poller starts writing "
            "events for a finished match this section populates automatically._"
        )
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Completed matches", n_completed)
        col2.metric("Log-loss", f"{wc2026_payload['log_loss']:.4f}")
        col3.metric("Brier", f"{wc2026_payload['brier']:.4f}")
        col4.metric("RPS", f"{wc2026_payload['rps']:.4f}")
        rows = [
            {
                "Date": row["match_date"],
                "Match": f"{row['home_team']} vs {row['away_team']}",
                "Score": f"{row['home_score']}-{row['away_score']}",
                "Observed": row["observed"],
                "P(H/D/A)": f"{row['p_home']:.0%} / {row['p_draw']:.0%} / {row['p_away']:.0%}",
                "Log-loss": round(row["log_loss"], 4),
                "Brier": round(row["brier"], 4),
                "RPS": round(row["rps"], 4),
                "Model": row["model_version"],
            }
            for row in wc2026_payload.get("per_match", [])
        ]
        st.dataframe(rows, hide_index=True, width="stretch")

st.divider()
st.subheader("Historical hindcasts (WC 2018 + WC 2022)")


@st.cache_data(show_spinner="Running WC hindcasts…", ttl=3600)
def run_hindcast(tournament: str) -> tuple[dict, list[dict]]:
    """Returns (overall_metrics, reliability_bins). Cached for the session.

    Heavy imports live at module top; the cache TTL keeps the actual hindcast work
    (~3s per tournament) from re-running per page visit.
    """
    from wc2026.eval.backtest import HindcastConfig, hindcast  # noqa: PLC0415
    from wc2026.eval.calibration import (  # noqa: PLC0415
        aggregate,
        base_rates,
        baseline_log_loss,
        reliability_diagram,
    )
    from wc2026.ingest.kaggle_intl import load_played  # noqa: PLC0415

    history = load_played()
    spans = {
        "WC 2022": (pd.Timestamp("2022-11-20"), pd.Timestamp("2022-12-18")),
        "WC 2018": (pd.Timestamp("2018-06-14"), pd.Timestamp("2018-07-15")),
    }
    start, end = spans[tournament]
    target = history[
        (history["tournament"] == "FIFA World Cup")
        & (history["date"] >= start)
        & (history["date"] <= end)
    ].copy()
    preds = hindcast(target, history, cfg=HindcastConfig())
    clean = preds.dropna(subset=["p_home", "p_draw", "p_away", "observed"])
    metrics = aggregate(clean)
    obs = clean["observed"].tolist()
    rates = base_rates(obs)
    return (
        {
            "log_loss": metrics.log_loss,
            "brier": metrics.brier,
            "rps": metrics.rps,
            "baseline_log_loss": baseline_log_loss(obs),
            "n": metrics.n,
            "base_h": rates["H"],
            "base_d": rates["D"],
            "base_a": rates["A"],
        },
        [
            {
                "outcome": b.outcome,
                "bin_low": b.bin_low,
                "bin_high": b.bin_high,
                "n": b.n,
                "mean_predicted": b.mean_predicted,
                "realized": b.realized_frequency,
            }
            for b in reliability_diagram(clean, n_bins=10)
            if b.n > 0
        ],
    )


tournament = st.radio("Tournament", options=("WC 2022", "WC 2018"), horizontal=True)

with st.spinner("Computing…"):
    metrics, bins = run_hindcast(tournament)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Matches", metrics["n"])
col2.metric(
    "Log-loss",
    f"{metrics['log_loss']:.4f}",
    delta=f"{metrics['log_loss'] - metrics['baseline_log_loss']:+.4f} vs base rates",
    delta_color="inverse",
)
col3.metric("Brier", f"{metrics['brier']:.4f}")
col4.metric("RPS", f"{metrics['rps']:.4f}")

st.caption(
    f"Base rates (climatological no-skill model): H={metrics['base_h']:.1%}, "
    f"D={metrics['base_d']:.1%}, A={metrics['base_a']:.1%}. "
    f"Climatological log-loss: {metrics['baseline_log_loss']:.4f}. "
    "Lower is better. Negative delta = we beat the no-skill baseline."
)

# Literature bookmaker baseline — closing odds are widely reported as the
# best publicly-available calibration target. football-data.co.uk (cited in
# the original blueprint) only covers Euros + domestic leagues; for World Cup
# bookmaker numbers we use published academic values instead of live ingest.
_BOOKMAKER_LITERATURE = {
    "WC 2018": {
        "log_loss_low": 0.96,
        "log_loss_high": 1.00,
        "cite": "Wheatcroft 2019 (RPS≈0.181); Constantinou 2019 (log-loss range)",
    },
    "WC 2022": {
        # No widely-cited closing-odds aggregate for WC 2022 in the open literature
        # as of this writing — these are conservative estimates from market consensus.
        "log_loss_low": 0.95,
        "log_loss_high": 1.00,
        "cite": "estimate from market consensus (no peer-reviewed aggregate yet)",
    },
}
_book = _BOOKMAKER_LITERATURE.get(tournament)
if _book is not None:
    delta_low = metrics["log_loss"] - _book["log_loss_high"]
    delta_high = metrics["log_loss"] - _book["log_loss_low"]
    sign = "+" if delta_low >= 0 else ""
    st.info(
        f"**Bookmaker reference** (closing-odds-implied probabilities): "
        f"log-loss ≈ **{_book['log_loss_low']:.2f}–{_book['log_loss_high']:.2f}**. "
        f"Our model trails by {sign}{delta_low:+.3f} to {sign}{delta_high:+.3f}. "
        f"Source: _{_book['cite']}_. Bookmaker odds incorporate injuries, news, "
        "and market signals our model cannot see — a 0.04–0.08 gap is the realistic ceiling."
    )

# Reliability diagram
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="perfect",
        line={"color": "#888", "dash": "dash"},
    )
)
colors = {"H": "#1f77b4", "D": "#7f7f7f", "A": "#d62728"}
names = {"H": "Home", "D": "Draw", "A": "Away"}
for outcome in ("H", "D", "A"):
    pts = [b for b in bins if b["outcome"] == outcome]
    if not pts:
        continue
    fig.add_trace(
        go.Scatter(
            x=[p["mean_predicted"] for p in pts],
            y=[p["realized"] for p in pts],
            mode="markers+lines",
            name=names[outcome],
            marker={"color": colors[outcome], "size": [max(8, p["n"]) for p in pts]},
        )
    )
fig.update_layout(
    xaxis_title="Predicted probability",
    yaxis_title="Realised frequency",
    xaxis={"range": [0, 1], "tickformat": ".0%"},
    yaxis={"range": [0, 1], "tickformat": ".0%"},
    height=500,
)
st.plotly_chart(fig, config=PLOTLY_CONFIG)

st.divider()
st.caption("Per-bin counts (size of marker above):")
st.dataframe(
    pd.DataFrame(bins).round(3),
    hide_index=True,
    width="stretch",
)
