"""Bracket Realisation — one sampled knockout outcome from the Monte Carlo.

This page is intentionally **not** an interactive bracket simulator. The blueprint
specified a click-to-set re-simulation UI; we decided not to build that (see
docs/methodology.md). Per-team round probabilities across all simulations are
shown on the Groups page; what you get here is one representative sample, picked
deterministically by the seed.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from dashboard.components.api_client import (
    APIUnreachable,
    get_json,
    render_unreachable_warning,
)

st.title("Knockout bracket (sample realisation)")

st.caption(
    "One representative sample from the Monte Carlo. Change the seed to draw a different "
    "realisation. For per-team **probabilities** across all simulations, see the Groups page."
)

seed = st.number_input("Seed", min_value=0, max_value=1_000_000, value=42, step=1)

try:
    data = get_json("/api/v1/tournament/bracket", params={"seed": int(seed)})
except APIUnreachable as exc:
    render_unreachable_warning(exc)
    st.stop()

st.success(f"🏆 Champion (seed {data['seed']}): **{data['champion']}**")

df = pd.DataFrame(data["matches"])
df["score"] = df["regulation_score"].apply(lambda s: f"{s[0]}–{s[1]}")
df["decided"] = df["decided_in"].map({"regulation": "90'", "extra_time": "AET", "shootout": "pens"})
df = df[["round", "match_id", "home_team", "score", "away_team", "winner", "decided"]]
df.columns = ["Round", "#", "Home", "Score", "Away", "Winner", "Decided"]

round_order = ["R32", "R16", "QF", "SF", "Final"]
for r in round_order:
    sub = df[df["Round"] == r]
    if sub.empty:
        continue
    with st.expander(f"{r} — {len(sub)} match(es)", expanded=(r in ("SF", "Final"))):
        st.dataframe(sub.drop(columns=["Round"]), hide_index=True, width="stretch")
