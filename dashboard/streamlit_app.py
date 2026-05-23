"""WC 2026 Predictions — Streamlit entrypoint.

The dashboard is a thin client over the FastAPI app at ``WC2026_API_URL``
(default ``http://localhost:8000``). Pages live under ``dashboard/pages/``.

Run locally:
    uv run uvicorn wc2026.api.main:app &
    uv run streamlit run dashboard/streamlit_app.py
"""

from __future__ import annotations

import os

import streamlit as st

API_URL = os.environ.get("WC2026_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="WC 2026 Predictions",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("FIFA World Cup 2026 — Predictions")

st.markdown(
    """
    Calibrated probabilistic forecasts for every match of the 2026 tournament,
    built on a weighted bivariate Poisson + Dixon–Coles model and an
    end-of-tournament Monte Carlo simulator.

    Use the **pages in the sidebar** to navigate:

    - **Today** — predictions for fixtures on the current/selected matchday
    - **Match Detail** — per-match probability + score heatmap + plain-language reasoning
    - **Groups** — group-stage advancement probabilities *(coming in Phase C.3)*
    - **Bracket** — knockout bracket and championship odds *(coming in Phase C.3)*
    - **Track Record** — historical calibration on WC 2018 / WC 2022 *(coming in Phase C.3)*
    """
)

with st.sidebar:
    st.subheader("About these forecasts")
    st.info(
        "**Probabilities are not certainties.** "
        "A 15% favourite still loses 85% of the time. "
        "The model trails published bookmaker closing odds by ~0.04–0.09 log-loss "
        "on high-upset tournaments — it cannot see injuries, news, or live market signal."
    )
    st.caption(f"API: `{API_URL}`")
    st.caption("Model: weighted Poisson + Dixon–Coles (Stage 0.6 tuned).")
