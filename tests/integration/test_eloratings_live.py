"""Integration test: actually hits eloratings.net.

Skipped by default. Run explicitly with:
    uv run pytest -m integration tests/integration/test_eloratings_live.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wc2026.ingest.eloratings_scraper import fetch_current_ratings

pytestmark = pytest.mark.integration


def test_live_fetch_writes_snapshot(tmp_path: Path) -> None:
    out = fetch_current_ratings(target_dir=tmp_path, cache_path=None)
    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) >= 200
    assert "team_name" in df.columns
    assert "rating" in df.columns
    # The site's top team is almost always ranked 1900+ Elo.
    assert df["rating"].max() >= 1900
