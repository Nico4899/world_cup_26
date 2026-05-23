"""WC 2026 Predictions — Streamlit entrypoint.

The dashboard is a thin client over the FastAPI app at ``WC2026_API_URL``
(default ``http://localhost:8000``). Pages live under ``dashboard/pages/``.

Run locally:
    uv run uvicorn wc2026.api.main:app &
    uv run streamlit run dashboard/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Streamlit prepends the script's own directory (dashboard/) to sys.path, not the
# project root, so `from dashboard.components.api_client import ...` in pages would
# raise ModuleNotFoundError. Insert the project root explicitly before any other
# imports so the package path resolves the same way pytest/uvicorn already use it.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

from dashboard.components.api_client import APIUnreachable, get_json  # noqa: E402

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

    - **Today** — prediction cards for the selected matchday
    - **Match Detail** — per-match probability + score heatmap + plain-language reasoning
    - **Groups** — group-stage advancement probabilities for all 12 groups
    - **Bracket Realisation** — one sampled knockout realisation (resampleable by seed)
    - **Track Record** — historical calibration on WC 2018 / WC 2022
    - **About** — model methodology, citations, and known limitations
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
    try:
        health = get_json("/health")
        fit_at = health.get("model_fit_at")
        version = health.get("model_version") or "unknown"
        fit_at_display = fit_at[:19].replace("T", " ") + " UTC" if fit_at else "unknown"
        st.caption(f"Model: `{version}` · fit at **{fit_at_display}**")
        # Surface eloratings snapshot staleness — warn if >7 days old, since
        # the shootout submodel keys off this snapshot.
        elo_age = health.get("elo_snapshot_age_days")
        elo_date = health.get("elo_snapshot_date")
        if elo_age is None:
            st.caption("Elo snapshot: _unavailable_ (shootouts fall back to 50/50).")
        elif elo_age > 7:
            st.warning(
                f"Elo snapshot is **{elo_age} days old** (captured {elo_date}). "
                "The daily ingest may not be running — check the scheduler logs."
            )
        else:
            st.caption(f"Elo snapshot: {elo_date} ({elo_age}d old).")
    except APIUnreachable:
        st.caption("Model: status unavailable (API not reachable).")
