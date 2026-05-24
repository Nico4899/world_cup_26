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


def test_get_match_by_id_includes_prediction_with_full_matrix(client: TestClient) -> None:
    r = client.get("/api/v1/matches/0")
    assert r.status_code == 200
    body = r.json()
    assert body["fixture"]["match_id"] == 0
    pred = body["prediction"]
    s = pred["outcome"]["home_win"] + pred["outcome"]["draw"] + pred["outcome"]["away_win"]
    assert abs(s - 1.0) < 1e-6
    # /api/v1/matches/{id} now returns top-5 + the full score_matrix in one call
    assert len(pred["top_scorelines"]) == 5
    matrix = pred["score_matrix"]
    assert matrix is not None
    assert len(matrix) == 11  # 0..10 goals (max_goals=10)
    assert all(len(row) == 11 for row in matrix)
    total_mass = sum(p for row in matrix for p in row)
    assert abs(total_mass - 1.0) < 1e-6


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
    # /api/v1/predictions/... ships the full score_matrix
    matrix = data["score_matrix"]
    assert matrix is not None
    assert len(matrix) == 11
    # The top-1 scoreline's matrix cell should equal its reported probability.
    top = data["top_scorelines"][0]
    assert abs(matrix[top["home_goals"]][top["away_goals"]] - top["probability"]) < 1e-9


def test_pairwise_prediction_422_on_unknown_team(client: TestClient) -> None:
    r = client.get("/api/v1/predictions/Atlantis/France")
    assert r.status_code == 422
    assert "Atlantis" in r.json()["detail"]


# --- /api/v1/tournament -----------------------------------------------------


def test_standings_returns_12_groups_with_per_team_probabilities(client: TestClient) -> None:
    # Use a small n_sims to keep the test fast.
    r = client.get("/api/v1/tournament/standings", params={"n_sims": 100, "seed": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["n_sims"] == 100
    assert len(body["groups"]) == 12
    for g in body["groups"]:
        assert len(g["teams"]) == 4
        for row in g["teams"]:
            for k in ("p_first", "p_second", "p_third_advance", "p_eliminated"):
                assert 0.0 <= row[k] <= 1.0
    # Headline has 10 teams sorted by p_champion descending
    assert len(body["headline"]) == 10
    ps = [t["p_champion"] for t in body["headline"]]
    assert ps == sorted(ps, reverse=True)


def test_bracket_returns_31_knockout_matches_and_champion(client: TestClient) -> None:
    r = client.get("/api/v1/tournament/bracket", params={"seed": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["seed"] == 0
    # 16 R32 + 8 R16 + 4 QF + 2 SF + 1 Final = 31 matches
    assert len(body["matches"]) == 31
    rounds = {m["round"] for m in body["matches"]}
    assert rounds == {"R32", "R16", "QF", "SF", "Final"}
    # Champion is one of the teams in the final
    final_match = next(m for m in body["matches"] if m["round"] == "Final")
    assert body["champion"] in (final_match["home_team"], final_match["away_team"])


def test_bracket_is_cached_by_seed(client: TestClient) -> None:
    """Two calls with the same seed must return byte-identical brackets (cache hit)."""
    r1 = client.get("/api/v1/tournament/bracket", params={"seed": 7})
    r2 = client.get("/api/v1/tournament/bracket", params={"seed": 7})
    assert r1.status_code == r2.status_code == 200
    assert r1.json() == r2.json()
    # different seed gives a different realisation (almost surely)
    r3 = client.get("/api/v1/tournament/bracket", params={"seed": 8})
    assert r3.json()["matches"] != r1.json()["matches"]


# --- /health (enriched) ----------------------------------------------------


def test_health_exposes_model_fit_at_and_version(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["model_fit_at"] is not None
    assert body["model_version"] == "poisson_dc.v1"
    # ISO-formatted UTC timestamp
    assert "T" in body["model_fit_at"]


def test_health_exposes_group_assignment_source(client: TestClient) -> None:
    """/health surfaces whether group letters came from the FIFA draw or from
    fixture-date clique ordering. Without the JSON override on disk it should
    report 'derived'."""
    body = client.get("/health").json()
    assert "group_assignment_source" in body
    # The JSON override file is not committed; fresh test env uses derivation.
    assert body["group_assignment_source"] == "derived"


def test_health_exposes_elo_snapshot_freshness(client: TestClient) -> None:
    """If the eloratings snapshot is on disk, /health should surface its date +
    age + the shootout-model loaded flag."""
    body = client.get("/health").json()
    # In CI without the snapshot on disk these would be None / False; in the
    # repo's test environment the cached snapshot exists.
    assert body["elo_snapshot_date"] is not None
    assert body["elo_snapshot_age_days"] >= 0
    assert body["shootout_model_loaded"] is True


# --- /api/v1/teams/{team}/recent -------------------------------------------


def test_recent_form_returns_n_matches_from_team_perspective(client: TestClient) -> None:
    r = client.get("/api/v1/teams/Argentina/recent", params={"n": 5})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 5
    # Sorted desc by date
    dates = [row["date"] for row in rows]
    assert dates == sorted(dates, reverse=True)
    for row in rows:
        assert row["result"] in {"W", "D", "L"}
        assert row["venue"] in {"home", "away", "neutral"}
        # Argentina is never its own opponent
        assert row["opponent"] != "Argentina"


def test_recent_form_unknown_team_returns_422(client: TestClient) -> None:
    r = client.get("/api/v1/teams/Atlantis/recent")
    assert r.status_code == 422
    assert "Atlantis" in r.json()["detail"]


# --- /api/v1/h2h/{team_a}/{team_b} -----------------------------------------


def test_h2h_returns_matches_between_two_teams(client: TestClient) -> None:
    r = client.get("/api/v1/h2h/Argentina/Brazil", params={"n": 10})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) > 0
    # Every row pairs exactly Argentina vs Brazil
    for row in rows:
        teams = {row["home_team"], row["away_team"]}
        assert teams == {"Argentina", "Brazil"}
    # Sorted desc by date
    dates = [row["date"] for row in rows]
    assert dates == sorted(dates, reverse=True)


def test_h2h_never_met_returns_empty_not_422(client: TestClient) -> None:
    # San Marino vs Vanuatu — extremely unlikely fixture; defensive empty-OK.
    r = client.get("/api/v1/h2h/San Marino/Vanuatu")
    assert r.status_code == 200
    assert r.json() == []


def test_h2h_unknown_team_returns_422(client: TestClient) -> None:
    r = client.get("/api/v1/h2h/Atlantis/Argentina")
    assert r.status_code == 422
    assert "Atlantis" in r.json()["detail"]


# --- /api/v1/_ops/scheduler-status -----------------------------------------


def test_model_ref_date_is_dynamic_not_hardcoded() -> None:
    """Regression: the API's lifespan fit-reference date must follow today's
    UTC date, not a constant pinned at module-import time."""
    from datetime import UTC, date, datetime

    import pandas as pd

    from wc2026.api.main import _today_utc_ts

    today = pd.Timestamp(datetime.now(UTC).date())
    got = _today_utc_ts()
    assert isinstance(got, pd.Timestamp)
    # Allow a small window in case the test crosses midnight UTC mid-run.
    assert got.date() in (today.date() - pd.Timedelta(days=1).to_pytimedelta(), today.date())
    # Must NOT be the old 2026-05-23 hardcode.
    assert got.date() != date(2026, 5, 23) or today.date() == date(2026, 5, 23)


def test_scheduler_status_returns_503_or_empty_without_db(client: TestClient) -> None:
    """In the unit-test env we have no Postgres. The endpoint must either:
    - return 503 (DB unreachable), or
    - return 200 with an empty job list (if a local test DB happens to exist
      with the table but no rows).
    What we MUST NOT do is crash the API or leak a stack trace.
    """
    r = client.get("/api/v1/_ops/scheduler-status")
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        body = r.json()
        assert "jobs" in body
        assert isinstance(body["jobs"], list)
    else:
        # 503 body must be informative without leaking internals.
        body = r.json()
        assert "Scheduler-status DB query failed" in body["detail"]


# --- /api/v1/_ops/available-jobs + /run-job --------------------------------


def test_available_jobs_lists_all_registered_jobs(client: TestClient) -> None:
    r = client.get("/api/v1/_ops/available-jobs")
    assert r.status_code == 200
    body = r.json()
    assert "jobs" in body
    # The Phase 2 additions plus the originals must all be there.
    expected_subset = {
        "db_backup",
        "kaggle_refresh",
        "elo_refresh",
        "poisson_refit",
        "thesportsdb_refresh",
        "openfootball_refresh",
        "fifa_ranking_refresh",
        "wikipedia_squads_refresh",
    }
    assert expected_subset.issubset(set(body["jobs"]))


def test_run_job_404_on_unknown_name(client: TestClient, monkeypatch) -> None:
    monkeypatch.delenv("WC2026_OPS_TOKEN", raising=False)
    r = client.post("/api/v1/_ops/run-job/nope")
    assert r.status_code == 404
    assert "unknown job" in r.json()["detail"]


def test_run_job_enqueues_known_job_when_token_unset(client: TestClient, monkeypatch) -> None:
    """With WC2026_OPS_TOKEN unset, manual runs are open and return 202."""
    from wc2026.api.routes import ops as ops_mod

    monkeypatch.delenv("WC2026_OPS_TOKEN", raising=False)
    monkeypatch.setattr(ops_mod, "_run_job_safely", lambda name: None)
    r = client.post("/api/v1/_ops/run-job/openfootball_refresh")
    assert r.status_code == 202
    body = r.json()
    assert body["job_name"] == "openfootball_refresh"
    assert body["status"] == "enqueued"


def test_run_job_rejects_bad_token_when_required(client: TestClient, monkeypatch) -> None:
    monkeypatch.setenv("WC2026_OPS_TOKEN", "secret")
    r = client.post(
        "/api/v1/_ops/run-job/openfootball_refresh",
        headers={"X-Ops-Token": "wrong"},
    )
    assert r.status_code == 403


def test_run_job_accepts_matching_token(client: TestClient, monkeypatch) -> None:
    from wc2026.api.routes import ops as ops_mod

    monkeypatch.setenv("WC2026_OPS_TOKEN", "secret")
    monkeypatch.setattr(ops_mod, "_run_job_safely", lambda name: None)
    r = client.post(
        "/api/v1/_ops/run-job/openfootball_refresh",
        headers={"X-Ops-Token": "secret"},
    )
    assert r.status_code == 202


# --- Phase 5 blend + /api/v1/explain ---------------------------------------


def _inject_xgb(client: TestClient):
    """Train a tiny XGB on synthetic feature rows and attach it to app.state.

    Returns ``(xgb_model, explainer)``; callers swap them in via monkeypatch
    so we don't have to wait for the real refit script (which is slow).
    """
    import numpy as np
    import pandas as pd

    from wc2026.models.shap_explain import XgbExplainer
    from wc2026.models.xgb_classifier import (
        CLASS_AWAY,
        CLASS_DRAW,
        CLASS_HOME,
        DEFAULT_FEATURE_COLUMNS,
        XgbMatchModel,
    )

    rng = np.random.default_rng(0)
    n = 600
    elo_diff = rng.normal(0.0, 120.0, size=n)
    base = {col: np.zeros(n) for col in DEFAULT_FEATURE_COLUMNS}
    base["elo_diff"] = elo_diff
    base["poisson_p_home"] = np.clip(0.4 + 0.0015 * elo_diff, 0.05, 0.9)
    base["poisson_p_draw"] = np.full(n, 0.27)
    base["poisson_p_away"] = 1 - base["poisson_p_home"] - base["poisson_p_draw"]
    X = pd.DataFrame(base)
    y = np.where(
        elo_diff > 60, CLASS_HOME, np.where(elo_diff < -60, CLASS_AWAY, CLASS_DRAW)
    ).astype(int)
    model = XgbMatchModel.fit(X, y)
    explainer = XgbExplainer.from_model(model)
    return model, explainer


def test_predictions_blend_no_op_without_xgb_artefact(client: TestClient) -> None:
    """With ``blend=true`` but no XGB loaded, the route returns Poisson-only."""
    r = client.get(
        "/api/v1/predictions/Argentina/France",
        params={"neutral": "true", "blend": "true"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["blend"] is None
    # Outcome still totals 1.
    s = body["outcome"]["home_win"] + body["outcome"]["draw"] + body["outcome"]["away_win"]
    assert abs(s - 1.0) < 1e-6


def test_predictions_blend_populates_when_xgb_loaded(client: TestClient, monkeypatch) -> None:
    xgb_model, explainer = _inject_xgb(client)
    monkeypatch.setattr(client.app.state, "xgb_model", xgb_model)
    monkeypatch.setattr(client.app.state, "xgb_explainer", explainer)
    r = client.get(
        "/api/v1/predictions/Argentina/France",
        params={"neutral": "true", "blend": "true", "blend_weight": 0.5},
    )
    assert r.status_code == 200
    body = r.json()
    blend = body["blend"]
    assert blend is not None
    assert blend["weight"] == 0.5
    for triplet_key in ("poisson", "xgb", "blended"):
        triplet = blend[triplet_key]
        s = triplet["home_win"] + triplet["draw"] + triplet["away_win"]
        assert abs(s - 1.0) < 1e-6
    # The response's headline outcome is now the blend (not Poisson-only).
    assert body["outcome"] == blend["blended"]


def test_predictions_blend_weight_one_returns_pure_poisson(client: TestClient, monkeypatch) -> None:
    xgb_model, explainer = _inject_xgb(client)
    monkeypatch.setattr(client.app.state, "xgb_model", xgb_model)
    monkeypatch.setattr(client.app.state, "xgb_explainer", explainer)
    r = client.get(
        "/api/v1/predictions/Argentina/France",
        params={"blend": "true", "blend_weight": 1.0},
    )
    assert r.status_code == 200
    body = r.json()
    blend = body["blend"]
    # weight=1 → blended must equal the Poisson component (up to FP).
    assert abs(blend["blended"]["home_win"] - blend["poisson"]["home_win"]) < 1e-6


def test_predictions_blend_weight_out_of_range_is_422(client: TestClient) -> None:
    r = client.get(
        "/api/v1/predictions/Argentina/France",
        params={"blend": "true", "blend_weight": 1.5},
    )
    assert r.status_code == 422


# --- /api/v1/explain/{match_id} --------------------------------------------


def test_explain_returns_503_without_xgb(client: TestClient) -> None:
    r = client.get("/api/v1/explain/0")
    assert r.status_code == 503
    assert "XGB" in r.json()["detail"]


def test_explain_returns_top_features_for_match_when_xgb_loaded(
    client: TestClient, monkeypatch
) -> None:
    xgb_model, explainer = _inject_xgb(client)
    monkeypatch.setattr(client.app.state, "xgb_model", xgb_model)
    monkeypatch.setattr(client.app.state, "xgb_explainer", explainer)
    r = client.get("/api/v1/explain/0", params={"top_n": 4})
    assert r.status_code == 200
    body = r.json()
    assert body["home_team"] == "Mexico"  # First fixture in the WC 2026 corpus
    assert len(body["contributions"]) == 4
    for item in body["contributions"]:
        assert "feature" in item
        assert "contribution" in item
    # Sorted by |contribution| desc
    abs_contribs = [abs(c["contribution"]) for c in body["contributions"]]
    assert abs_contribs == sorted(abs_contribs, reverse=True)
    # Both poisson + xgb outcomes are populated.
    assert body["poisson_outcome"] is not None
    assert body["xgb_outcome"] is not None


def test_explain_rejects_unknown_class_name(client: TestClient, monkeypatch) -> None:
    xgb_model, explainer = _inject_xgb(client)
    monkeypatch.setattr(client.app.state, "xgb_model", xgb_model)
    monkeypatch.setattr(client.app.state, "xgb_explainer", explainer)
    r = client.get("/api/v1/explain/0", params={"class_name": "victory_dance"})
    assert r.status_code == 422


def test_explain_404_on_out_of_range_match_id(client: TestClient, monkeypatch) -> None:
    xgb_model, explainer = _inject_xgb(client)
    monkeypatch.setattr(client.app.state, "xgb_model", xgb_model)
    monkeypatch.setattr(client.app.state, "xgb_explainer", explainer)
    r = client.get("/api/v1/explain/999")
    assert r.status_code == 404
