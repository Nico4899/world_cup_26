"""About — render the methodology document inside the dashboard.

The methodology lives at docs/methodology.md (single source of truth, also
versioned in git). Surfacing it here makes the model's reasoning visible to
end users without leaving the dashboard — directly serves the "calibrated
honesty" framing.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_METHODOLOGY_PATH = Path(__file__).resolve().parent.parent.parent / "docs" / "methodology.md"

st.title("About / Methodology")

if not _METHODOLOGY_PATH.exists():
    st.error(f"`{_METHODOLOGY_PATH}` not found in this checkout.")
    st.stop()

st.markdown(_METHODOLOGY_PATH.read_text(encoding="utf-8"))
