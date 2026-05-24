"""Shared Plotly chart config for every dashboard page.

The default Plotly modebar gives PNG download for free, which the dashboard
spec calls out as a baseline expectation ("Every chart must support download
as PNG"). We keep the toolbar but trim the buttons we don't want (zoom/pan
add nothing to small probability bars and clutter the UI).

Apply as ``st.plotly_chart(fig, config=PLOTLY_CONFIG)``.
"""

from __future__ import annotations

# Buttons we don't want surfaced for the kinds of charts the dashboard uses
# (small probability stacks, single-line win-prob timelines). The download
# button stays — that's the whole point.
_TRIMMED_BUTTONS = [
    "zoom2d",
    "pan2d",
    "select2d",
    "lasso2d",
    "autoScale2d",
    "resetScale2d",
    "zoomIn2d",
    "zoomOut2d",
]

PLOTLY_CONFIG: dict = {
    "displaylogo": False,
    "modeBarButtonsToRemove": _TRIMMED_BUTTONS,
    "toImageButtonOptions": {
        "format": "png",
        "scale": 2,
        "filename": "wc2026-chart",
    },
}


__all__ = ["PLOTLY_CONFIG"]
