"""About — render the methodology document inside the dashboard.

The methodology lives at docs/methodology.md (single source of truth, also
versioned in git). Surfacing it here makes the model's reasoning visible to
end users without leaving the dashboard — directly serves the "calibrated
honesty" framing.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from dashboard.components.api_client import APIUnreachable, get_json

_METHODOLOGY_PATH = Path(__file__).resolve().parent.parent.parent / "docs" / "methodology.md"

st.title("About / Methodology")

# Surface where the group-letter assignment came from so users can tell whether
# the labels are the official FIFA draw or our fixture-date-based derivation.
try:
    _health = get_json("/health")
    _source = _health.get("group_assignment_source", "derived")
    if _source == "derived":
        st.info(
            "**Group letters A–L** are currently *derived* from the order of fixture "
            "dates (Group A = the earliest opener, and so on). They may not match "
            "FIFA's official draw letters until "
            "`data/wc2026_group_assignment.json` is populated."
        )
    else:
        st.success(f"**Group letters A–L** sourced from {_source.removeprefix('official:')}.")
except APIUnreachable:
    pass

if not _METHODOLOGY_PATH.exists():
    st.error(f"`{_METHODOLOGY_PATH}` not found in this checkout.")
    st.stop()

st.markdown(_METHODOLOGY_PATH.read_text(encoding="utf-8"))
