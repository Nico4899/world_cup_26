"""Integration test: hit the live football-data.org API if a key is configured.

Run with:
    FOOTBALL_DATA_ORG_KEY=... uv run pytest -m integration tests/integration/test_football_data_org_live.py
"""

from __future__ import annotations

import os

import pytest

from wc2026.ingest import football_data_org as fdo

pytestmark = pytest.mark.integration


@pytest.fixture
def require_key() -> str:
    key = os.environ.get(fdo.ENV_API_KEY)
    if not key:
        pytest.skip(f"{fdo.ENV_API_KEY} not set — skipping live API test")
    return key


def test_fetch_world_cup_matches_returns_nonempty_dataframe(require_key):
    df = fdo.fetch_competition_matches(fdo.WC_COMPETITION_CODE, api_key=require_key)
    assert not df.empty
    assert "home_team" in df.columns
    assert "utc_date" in df.columns
