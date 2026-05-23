"""Tests for the FastAPI app routes (no external deps; uses TestClient + lifespan)."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from wc2026.api.main import app


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """One TestClient per module — lifespan-loaded model + fixtures stay cached."""
    with TestClient(app) as c:
        yield c


# --- /health ----------------------------------------------------------------


def test_health_returns_ok_with_loaded_model(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_fitted"] is True
    # ~291 teams from the 10-year fit; allow generous range so test isn't brittle.
    assert data["model_teams_n"] > 100


# --- /api/v1/matches --------------------------------------------------------


def test_list_matches_returns_72_wc_2026_fixtures(client: TestClient) -> None:
    r = client.get("/api/v1/matches")
    assert r.status_code == 200
    matches = r.json()
    assert len(matches) == 72
    first = matches[0]
    assert first["match_id"] == 0
    assert first["home_team"] == "Mexico"
    assert first["group"] == "A"


def test_list_matches_filter_by_group(client: TestClient) -> None:
    r = client.get("/api/v1/matches", params={"group": "A"})
    assert r.status_code == 200
    matches = r.json()
    # group A has 4 teams → 6 matches
    assert len(matches) == 6
    assert all(m["group"] == "A" for m in matches)


def test_list_matches_filter_by_date(client: TestClient) -> None:
    r = client.get("/api/v1/matches", params={"date": "2026-06-11"})
    assert r.status_code == 200
    matches = r.json()
    # June 11 is the opener: 2 matches (both Group A)
    assert len(matches) == 2
    assert all(m["date"] == "2026-06-11" for m in matches)


def test_get_match_by_id_includes_prediction(client: TestClient) -> None:
    r = client.get("/api/v1/matches/0")
    assert r.status_code == 200
    body = r.json()
    assert body["fixture"]["match_id"] == 0
    pred = body["prediction"]
    s = pred["outcome"]["home_win"] + pred["outcome"]["draw"] + pred["outcome"]["away_win"]
    assert abs(s - 1.0) < 1e-6
    # top_n=3 for the by-id endpoint
    assert len(pred["top_scorelines"]) == 3


def test_get_match_by_id_404_on_out_of_range(client: TestClient) -> None:
    r = client.get("/api/v1/matches/999")
    assert r.status_code == 404


# --- /api/v1/predictions ----------------------------------------------------


def test_pairwise_prediction_known_teams(client: TestClient) -> None:
    r = client.get("/api/v1/predictions/Argentina/France", params={"neutral": "true"})
    assert r.status_code == 200
    data = r.json()
    s = data["outcome"]["home_win"] + data["outcome"]["draw"] + data["outcome"]["away_win"]
    assert abs(s - 1.0) < 1e-6
    assert data["expected_home_goals"] > 0
    assert data["expected_away_goals"] > 0
    assert len(data["top_scorelines"]) == 5
    # top scoreline probabilities should be monotonically non-increasing
    probs = [sc["probability"] for sc in data["top_scorelines"]]
    assert probs == sorted(probs, reverse=True)


def test_pairwise_prediction_422_on_unknown_team(client: TestClient) -> None:
    r = client.get("/api/v1/predictions/Atlantis/France")
    assert r.status_code == 422
    assert "Atlantis" in r.json()["detail"]
