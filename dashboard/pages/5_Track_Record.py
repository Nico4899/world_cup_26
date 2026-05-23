"""Track Record — reliability diagrams + headline metrics on past tournaments.

Runs the WC 2022 / WC 2018 hindcasts inline (cached for the session) and plots
per-outcome reliability. This is the calibration honesty page: the model's
trustworthiness lives or dies here.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.title("Track record")

st.caption(
    "Calibration check on completed World Cups (out-of-sample). Each point is one "
    "predicted-probability bin; the dashed line is perfect calibration."
)


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
st.plotly_chart(fig, config={"displayModeBar": False})

st.divider()
st.caption("Per-bin counts (size of marker above):")
st.dataframe(
    pd.DataFrame(bins).round(3),
    hide_index=True,
    width="stretch",
)
