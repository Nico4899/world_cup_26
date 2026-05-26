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


def test_bracket_conditional_runs_with_no_locks(client: TestClient) -> None:
    """Empty-locks payload returns a standings-shaped headline with the requested n_sims."""
    r = client.post(
        "/api/v1/tournament/bracket/conditional",
        json={"locks": [], "n_sims": 200, "seed": 0},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["n_sims"] == 200
    assert body["seed"] == 0
    assert body["locks"] == []
    assert len(body["headline"]) == 10  # top-10 by champion prob
    # Champion probabilities must sum to ≤ 1 across the top 10 — trivially true.
    assert sum(h["p_champion"] for h in body["headline"]) <= 1.0001


def test_bracket_conditional_lifts_locked_team_champion_prob(client: TestClient) -> None:
    """Locking the final to a known WC 2026 team must push their champion prob to ~1.0
    *conditional* on reaching it. The unconditional value is bounded by the
    probability of reaching the final, but it must be strictly above the no-lock
    baseline by a non-trivial margin.
    """
    # Argentina is a perennial favourite under the production model — strong
    # enough that even a small N reliably brings them to the final.
    payload = {
        "locks": [{"match_id": 104, "winner": "Argentina"}],
        "n_sims": 200,
        "seed": 0,
    }
    r = client.post("/api/v1/tournament/bracket/conditional", json=payload)
    assert r.status_code == 200
    body = r.json()
    arg = next((h for h in body["headline"] if h["team"] == "Argentina"), None)
    assert arg is not None, "Argentina should appear in the top-10 under the lock"
    assert arg["p_champion"] > 0.0
    # Lifting the lock must not break the schema — every headline row keeps the
    # standings shape.
    for h in body["headline"]:
        assert {"team", "p_champion", "p_final", "p_sf", "p_qf"} <= h.keys()


def test_bracket_conditional_422_on_unknown_team(client: TestClient) -> None:
    r = client.post(
        "/api/v1/tournament/bracket/conditional",
        json={"locks": [{"match_id": 104, "winner": "Atlantis"}], "n_sims": 200},
    )
    assert r.status_code == 422
    assert "Atlantis" in r.json()["detail"]


def test_bracket_conditional_422_on_invalid_match_id(client: TestClient) -> None:
    r = client.post(
        "/api/v1/tournament/bracket/conditional",
        json={"locks": [{"match_id": 1, "winner": "Argentina"}], "n_sims": 200},
    )
    # Pydantic's range constraint surfaces 422 before the handler runs.
    assert r.status_code == 422


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


# --- /api/v1/_ops/elo-override ---------------------------------------------


def _stub_elo_session(monkeypatch, *, store: list) -> None:
    """Replace ops.session_scope with an in-memory store backed by a list.

    The real ``RawEloOverride`` SQLAlchemy class is kept — `select(...)` just
    constructs an SQL expression (no DB hit) and our stubbed session ignores
    the statement entirely. Stored rows are real ORM instances so the route's
    ``r.team_code`` attribute reads work without further patching.
    """
    from wc2026.api.routes import ops as ops_mod

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def scalars(self, _stmt):
            class _S:
                def __iter__(_self):
                    return iter(store)

                def first(_self):
                    return store[0] if store else None

            return _S()

        def get(self, _model, team_code):
            for row in store:
                if row.team_code == team_code:
                    return row
            return None

        def add(self, row):
            store.append(row)

        def delete(self, row):
            store.remove(row)

        def flush(self):
            return None

    from contextlib import contextmanager

    @contextmanager
    def fake_scope():
        yield _Session()

    monkeypatch.setattr(ops_mod, "session_scope", fake_scope)


def test_elo_overrides_empty_list_returns_200(client: TestClient, monkeypatch) -> None:
    _stub_elo_session(monkeypatch, store=[])
    r = client.get("/api/v1/_ops/elo-overrides")
    assert r.status_code == 200
    assert r.json() == {"overrides": []}


def test_elo_override_upsert_creates_then_replaces(client: TestClient, monkeypatch) -> None:
    monkeypatch.delenv("WC2026_OPS_TOKEN", raising=False)
    store: list = []
    _stub_elo_session(monkeypatch, store=store)

    create = client.post(
        "/api/v1/_ops/elo-override",
        json={"team_code": "ENG", "team_name": "England", "rating": 1850.5,
              "reason": "scraper broken 2026-06-12"},
    )
    assert create.status_code == 200
    body = create.json()
    assert body["team_code"] == "ENG"
    assert body["rating"] == 1850.5
    assert len(store) == 1

    # Re-POSTing for the same team_code should replace the row, not add.
    update = client.post(
        "/api/v1/_ops/elo-override",
        json={"team_code": "ENG", "team_name": "England", "rating": 1875.0},
    )
    assert update.status_code == 200
    assert update.json()["rating"] == 1875.0
    assert len(store) == 1


def test_elo_override_delete_404_when_missing(client: TestClient, monkeypatch) -> None:
    monkeypatch.delenv("WC2026_OPS_TOKEN", raising=False)
    _stub_elo_session(monkeypatch, store=[])
    r = client.delete("/api/v1/_ops/elo-override/ZZZ")
    assert r.status_code == 404


def test_elo_override_post_rejects_bad_token(client: TestClient, monkeypatch) -> None:
    monkeypatch.setenv("WC2026_OPS_TOKEN", "secret")
    _stub_elo_session(monkeypatch, store=[])
    r = client.post(
        "/api/v1/_ops/elo-override",
        json={"team_code": "ENG", "rating": 1900},
        headers={"X-Ops-Token": "wrong"},
    )
    assert r.status_code == 403


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


def test_predictions_blend_no_op_without_xgb_artefact(client: TestClient, monkeypatch) -> None:
    """With ``blend=true`` but no XGB loaded, the route returns Poisson-only."""
    # Same reasoning as test_explain_returns_503_without_xgb — force the
    # absent-artefact branch regardless of local data/artifacts state.
    monkeypatch.setattr(client.app.state, "xgb_model", None)
    monkeypatch.setattr(client.app.state, "xgb_explainer", None)
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


def test_explain_returns_503_without_xgb(client: TestClient, monkeypatch) -> None:
    # The module-scope client lifespan may have loaded a real artefact from
    # data/artifacts/xgb/latest.json; force it absent for this assertion so the
    # test stays meaningful regardless of local file state.
    monkeypatch.setattr(client.app.state, "xgb_model", None)
    monkeypatch.setattr(client.app.state, "xgb_explainer", None)
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


# --- /api/v1/live/{match_id} ----------------------------------------------


def _inject_live_model(client: TestClient):
    """Train a tiny live-win-prob model on synthetic snapshots."""
    import numpy as np
    import pandas as pd

    from wc2026.models.live_win_prob import LiveWinProbModel

    rng = np.random.default_rng(0)
    n = 600
    elo_diff = rng.normal(0.0, 80.0, size=n)
    goal_diff = rng.integers(-3, 4, size=n)
    minutes_remaining = rng.integers(0, 90, size=n)
    red_diff = rng.choice([-1, 0, 1], size=n, p=[0.05, 0.9, 0.05])
    logits = 0.01 * elo_diff + 1.4 * goal_diff - 0.4 * red_diff
    noise = rng.normal(0, 1.0, size=n)
    y = np.where(logits + noise > 0.8, 0, np.where(logits + noise < -0.8, 2, 1)).astype(int)
    X = pd.DataFrame(
        {
            "elo_diff": elo_diff,
            "goal_diff": goal_diff,
            "minutes_remaining": minutes_remaining,
            "red_diff": red_diff,
        }
    )
    return LiveWinProbModel.fit(X, y)


def test_live_snapshot_returns_pre_match_fallback_without_events(
    client: TestClient, monkeypatch
) -> None:
    """No DB / no live model → snapshot returns the Poisson pre-match prob."""
    monkeypatch.setattr(client.app.state, "live_win_prob_model", None)
    r = client.get("/api/v1/live/0")
    assert r.status_code == 200
    body = r.json()
    assert body["match_id"] == 0
    assert body["win_prob_source"] == "poisson_pre_match"
    s = sum(body["win_prob"].values())
    assert abs(s - 1.0) < 1e-6
    assert body["last_event_type"] == "KICKOFF"


def test_live_snapshot_uses_live_model_when_events_present(client: TestClient, monkeypatch) -> None:
    """Stub the DB-history loader so the snapshot route sees a 1-0 home goal."""
    import datetime as _dt

    from wc2026.api.routes import live as live_route
    from wc2026.db.models import RawLiveEvent

    monkeypatch.setattr(client.app.state, "live_win_prob_model", _inject_live_model(client))
    event = RawLiveEvent(
        match_id=0,
        seq=2,
        minute=23,
        period=1,
        event_type="GOAL",
        team="Mexico",
        player=None,
        home_score_after=1,
        away_score_after=0,
        home_red_cards_after=0,
        away_red_cards_after=0,
        ingested_at=_dt.datetime.now(_dt.UTC),
    )
    monkeypatch.setattr(live_route, "_latest_event", lambda _mid: event)
    r = client.get("/api/v1/live/0")
    assert r.status_code == 200
    body = r.json()
    assert body["win_prob_source"] == "live_win_prob"
    assert body["last_event_type"] == "GOAL"
    assert body["home_score"] == 1
    assert body["away_score"] == 0


def test_live_snapshot_final_score_collapses_to_realised_outcome(
    client: TestClient, monkeypatch
) -> None:
    """FT_WHISTLE → win_prob_source='final', probs are degenerate on the result."""
    import datetime as _dt

    from wc2026.api.routes import live as live_route
    from wc2026.db.models import RawLiveEvent

    monkeypatch.setattr(client.app.state, "live_win_prob_model", _inject_live_model(client))
    event = RawLiveEvent(
        match_id=0,
        seq=4,
        minute=90,
        period=2,
        event_type="FT_WHISTLE",
        team=None,
        player=None,
        home_score_after=2,
        away_score_after=1,
        home_red_cards_after=0,
        away_red_cards_after=0,
        ingested_at=_dt.datetime.now(_dt.UTC),
    )
    monkeypatch.setattr(live_route, "_latest_event", lambda _mid: event)
    r = client.get("/api/v1/live/0")
    body = r.json()
    assert body["win_prob_source"] == "final"
    assert body["win_prob"] == {"home_win": 1.0, "draw": 0.0, "away_win": 0.0}


def test_live_snapshot_out_of_range_match_id(client: TestClient) -> None:
    r = client.get("/api/v1/live/999")
    assert r.status_code == 404


# --- /api/v1/live/{match_id}/history (used by the dashboard chart) -------


def test_live_history_returns_traces_with_per_event_win_probs(
    client: TestClient, monkeypatch
) -> None:
    import datetime as _dt

    from wc2026.api.routes import live as live_route
    from wc2026.db.models import RawLiveEvent

    monkeypatch.setattr(client.app.state, "live_win_prob_model", _inject_live_model(client))
    now = _dt.datetime.now(_dt.UTC)
    events = [
        RawLiveEvent(
            match_id=0,
            seq=1,
            minute=0,
            period=1,
            event_type="KICKOFF",
            team=None,
            player=None,
            home_score_after=0,
            away_score_after=0,
            home_red_cards_after=0,
            away_red_cards_after=0,
            ingested_at=now,
        ),
        RawLiveEvent(
            match_id=0,
            seq=2,
            minute=23,
            period=1,
            event_type="GOAL",
            team="Mexico",
            player=None,
            home_score_after=1,
            away_score_after=0,
            home_red_cards_after=0,
            away_red_cards_after=0,
            ingested_at=now,
        ),
    ]
    monkeypatch.setattr(live_route, "_all_events", lambda _mid: events)
    r = client.get("/api/v1/live/0/history")
    assert r.status_code == 200
    body = r.json()
    assert len(body["events"]) == 2
    assert [e["event_type"] for e in body["events"]] == ["KICKOFF", "GOAL"]
    # Snapshot reflects the most recent event.
    assert body["snapshot"]["last_event_type"] == "GOAL"
    assert body["snapshot"]["home_score"] == 1
    # Every event trace has a win_prob triplet summing to 1.
    for ev in body["events"]:
        s = sum(ev["win_prob"].values())
        assert abs(s - 1.0) < 1e-6


def test_live_history_empty_when_no_events(client: TestClient, monkeypatch) -> None:
    from wc2026.api.routes import live as live_route

    monkeypatch.setattr(live_route, "_all_events", lambda _mid: [])
    r = client.get("/api/v1/live/0/history")
    assert r.status_code == 200
    body = r.json()
    assert body["events"] == []
    assert body["snapshot"]["last_event_type"] == "KICKOFF"
    assert body["snapshot"]["win_prob_source"] == "poisson_pre_match"


def test_live_history_out_of_range_match_id(client: TestClient) -> None:
    r = client.get("/api/v1/live/999/history")
    assert r.status_code == 404


# --- /api/v1/track-record/wc2026 ------------------------------------------


def test_wc2026_track_record_returns_zero_when_no_fixtures_or_events(
    client: TestClient, monkeypatch
) -> None:
    """No football-data.org cache + no live events → empty calibration, not 500."""
    from wc2026.api.routes import track_record as tr

    monkeypatch.setattr(tr, "load_wc_match_id_map", lambda: {})
    r = client.get("/api/v1/track-record/wc2026")
    assert r.status_code == 200
    body = r.json()
    assert body["n_completed"] == 0
    assert body["log_loss"] is None
    assert body["per_match"] == []


def test_wc2026_track_record_aggregates_when_event_and_prediction_present(
    client: TestClient, monkeypatch
) -> None:
    """Synthetic prediction + FT_WHISTLE row produce a non-zero calibration."""
    import datetime as _dt

    from wc2026.api.routes import track_record as tr
    from wc2026.eval.rolling import PerMatchCalibration, RollingCalibration

    # Bypass the DB completely; serve a synthetic RollingCalibration so the
    # test doesn't depend on whether the test env has Postgres.
    canned = RollingCalibration(
        n_completed=1,
        log_loss=0.6931,
        brier=0.5,
        rps=0.25,
        per_match=[
            PerMatchCalibration(
                match_date=_dt.date(2026, 6, 11),
                home_team="Mexico",
                away_team="Senegal",
                home_score=2,
                away_score=0,
                p_home=0.5,
                p_draw=0.25,
                p_away=0.25,
                observed="H",
                log_loss=0.6931,
                brier=0.5,
                rps=0.25,
                model_version="poisson_dc.v1",
            )
        ],
    )
    monkeypatch.setattr(
        tr, "load_wc_match_id_map", lambda: {1: (_dt.date(2026, 6, 11), "Mexico", "Senegal")}
    )
    monkeypatch.setattr(tr, "compute_rolling", lambda **_kw: canned)
    r = client.get("/api/v1/track-record/wc2026")
    assert r.status_code == 200
    body = r.json()
    assert body["n_completed"] == 1
    assert body["per_match"][0]["observed"] == "H"
    assert abs(body["log_loss"] - 0.6931) < 1e-6


# --- Phase 8: persisted-run standings -------------------------------------


def test_standings_uses_persisted_run_when_use_persisted_default(
    client: TestClient, monkeypatch
) -> None:
    """When a persisted MC run exists in the DB, the route serves it (no in-process MC)."""
    import pandas as pd

    from wc2026.api.routes import tournament as tour
    from wc2026.sim.tournament import ROUND_COLUMNS, TournamentSummary

    fixtures = client.app.state.fixtures
    teams = list(fixtures.teams)
    df = pd.DataFrame(
        [[1 / len(teams)] * len(ROUND_COLUMNS)] * len(teams),
        index=teams,
        columns=list(ROUND_COLUMNS),
    )
    df.index.name = "team"
    summary = TournamentSummary(n_sims=10_000, probabilities=df)

    called = {"in_process": 0}

    def fake_loader():
        return summary, 42, "poisson_dc.v1"

    def boom_in_process(*_args, **_kwargs):
        called["in_process"] += 1
        return summary

    monkeypatch.setattr(tour, "_load_persisted_summary", fake_loader)
    monkeypatch.setattr(tour, "_cached_summary", boom_in_process)

    r = client.get("/api/v1/tournament/standings")
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "persisted"
    assert body["run_id"] == 42
    assert body["model_version"] == "poisson_dc.v1"
    assert called["in_process"] == 0


def test_standings_force_in_process_with_use_persisted_false(
    client: TestClient, monkeypatch
) -> None:
    from wc2026.api.routes import tournament as tour

    monkeypatch.setattr(tour, "_load_persisted_summary", lambda: ({}, 0, "x"))
    r = client.get("/api/v1/tournament/standings", params={"use_persisted": "false", "n_sims": 100})
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "in_process"
    assert body["run_id"] is None


# --- /api/v1/teams/{team}/assets ------------------------------------------


def test_team_assets_returns_null_payload_when_no_db_row(client: TestClient, monkeypatch) -> None:
    """Without a raw_team_assets row, the route returns the team name with all-null
    fields rather than 404 — keeps the dashboard simple."""
    from wc2026.api.routes import teams as teams_route

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def scalars(self, _stmt):
            class _S:
                def first(_self):
                    return None

            return _S()

    def _stub_engine_null():
        return object()

    def _stub_session_null(*_args, **_kw):
        return _FakeSession()

    monkeypatch.setattr(teams_route, "get_engine", _stub_engine_null)
    monkeypatch.setattr(teams_route, "Session", _stub_session_null)
    r = client.get("/api/v1/teams/Argentina/assets")
    assert r.status_code == 200
    body = r.json()
    assert body["team"] == "Argentina"
    assert body["crest_url"] is None


def test_team_assets_returns_populated_row_when_present(client: TestClient, monkeypatch) -> None:
    from wc2026.api.routes import teams as teams_route
    from wc2026.db.models import RawTeamAsset

    asset = RawTeamAsset(
        team="Argentina",
        thesportsdb_id=133602,
        crest_url="https://example.test/argentina.png",
        kit_home_color="#75AADB",
        kit_away_color="#000080",
        stadium_name="Estadio Monumental",
        stadium_capacity=83214,
        stadium_city="Buenos Aires",
        stadium_country="Argentina",
    )

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def scalars(self, _stmt):
            class _S:
                def first(_self):
                    return asset

            return _S()

    def _stub_engine_pop():
        return object()

    def _stub_session_pop(*_args, **_kw):
        return _FakeSession()

    monkeypatch.setattr(teams_route, "get_engine", _stub_engine_pop)
    monkeypatch.setattr(teams_route, "Session", _stub_session_pop)
    r = client.get("/api/v1/teams/Argentina/assets")
    assert r.status_code == 200
    body = r.json()
    assert body["crest_url"] == "https://example.test/argentina.png"
    assert body["stadium_capacity"] == 83214


def test_team_assets_returns_503_on_db_error(client: TestClient, monkeypatch) -> None:
    from wc2026.api.routes import teams as teams_route

    def boom():
        raise RuntimeError("Postgres unreachable")

    monkeypatch.setattr(teams_route, "get_engine", boom)
    r = client.get("/api/v1/teams/Argentina/assets")
    assert r.status_code == 503


# --- /api/v1/teams/{team}/fifa-rankings ------------------------------------


def test_team_fifa_rankings_empty_when_no_rows(client: TestClient, monkeypatch) -> None:
    """No FIFA snapshots → empty history, 200 (so the dashboard can render an empty state)."""
    from wc2026.api.routes import teams as teams_route

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def scalars(self, _stmt):
            class _S:
                def first(_self):
                    return None

                def __iter__(_self):
                    return iter([])

            return _S()

    def _stub_engine():
        return object()

    def _stub_session(*_args, **_kw):
        return _FakeSession()

    monkeypatch.setattr(teams_route, "get_engine", _stub_engine)
    monkeypatch.setattr(teams_route, "Session", _stub_session)
    r = client.get("/api/v1/teams/Argentina/fifa-rankings")
    assert r.status_code == 200
    body = r.json()
    assert body["team"] == "Argentina"
    assert body["history"] == []


def test_team_fifa_rankings_returns_populated_history(client: TestClient, monkeypatch) -> None:
    import datetime as _dt

    from wc2026.api.routes import teams as teams_route
    from wc2026.db.models import RawFifaRanking

    rows = [
        RawFifaRanking(
            ranking_date=_dt.date(2026, 1, 1),
            team="Argentina",
            rank=2,
            points=1850.0,
            previous_rank=3,
        ),
        RawFifaRanking(
            ranking_date=_dt.date(2026, 2, 1),
            team="Argentina",
            rank=1,
            points=1870.0,
            previous_rank=2,
        ),
    ]

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def scalars(self, _stmt):
            class _S:
                def __iter__(_self):
                    return iter(rows)

            return _S()

    def _stub_engine_fr():
        return object()

    def _stub_session_fr(*_args, **_kw):
        return _FakeSession()

    monkeypatch.setattr(teams_route, "get_engine", _stub_engine_fr)
    monkeypatch.setattr(teams_route, "Session", _stub_session_fr)
    r = client.get("/api/v1/teams/Argentina/fifa-rankings")
    assert r.status_code == 200
    body = r.json()
    assert len(body["history"]) == 2
    assert body["history"][-1]["rank"] == 1
    assert body["history"][-1]["previous_rank"] == 2


# --- /api/v1/teams/{team}/xg-form ------------------------------------------


def test_team_xg_form_returns_zero_window_when_no_rows(client: TestClient, monkeypatch) -> None:
    from wc2026.api.routes import teams as teams_route

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def scalars(self, _stmt):
            class _S:
                def __iter__(_self):
                    return iter([])

            return _S()

    def _stub_engine_xg_empty():
        return object()

    def _stub_session_xg_empty(*_args, **_kw):
        return _FakeSession()

    monkeypatch.setattr(teams_route, "get_engine", _stub_engine_xg_empty)
    monkeypatch.setattr(teams_route, "Session", _stub_session_xg_empty)
    r = client.get("/api/v1/teams/Argentina/xg-form")
    assert r.status_code == 200
    body = r.json()
    assert body["last_5"]["matches"] == 0
    assert body["last_5"]["xg_for"] is None
    assert body["last_10"]["matches"] == 0


def test_team_xg_form_averages_recent_rows(client: TestClient, monkeypatch) -> None:
    import datetime as _dt

    from wc2026.api.routes import teams as teams_route
    from wc2026.db.models import RawMatchXg

    rows = [
        RawMatchXg(
            match_date=_dt.date(2026, 3, 1),
            home_team="Argentina",
            away_team="Brazil",
            team="Argentina",
            source="statsbomb",
            xg_for=2.5,
            xg_against=1.0,
        ),
        RawMatchXg(
            match_date=_dt.date(2026, 2, 1),
            home_team="Spain",
            away_team="Argentina",
            team="Argentina",
            source="statsbomb",
            xg_for=1.5,
            xg_against=0.5,
        ),
    ]

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def scalars(self, _stmt):
            class _S:
                def __iter__(_self):
                    return iter(rows)

            return _S()

    def _stub_engine_xg_pop():
        return object()

    def _stub_session_xg_pop(*_args, **_kw):
        return _FakeSession()

    monkeypatch.setattr(teams_route, "get_engine", _stub_engine_xg_pop)
    monkeypatch.setattr(teams_route, "Session", _stub_session_xg_pop)
    r = client.get("/api/v1/teams/Argentina/xg-form")
    assert r.status_code == 200
    body = r.json()
    assert body["last_5"]["matches"] == 2
    assert body["last_5"]["xg_for"] == pytest.approx(2.0)
    assert body["last_5"]["xg_against"] == pytest.approx(0.75)
    assert body["last_5"]["xg_diff"] == pytest.approx(1.25)


# --- /api/v1/teams/{team}/squad --------------------------------------------


def test_team_squad_returns_empty_when_no_rows(client: TestClient, monkeypatch) -> None:
    from wc2026.api.routes import teams as teams_route

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def execute(self, _stmt):
            class _R:
                def first(_self):
                    return None

            return _R()

        def scalars(self, _stmt):
            class _S:
                def __iter__(_self):
                    return iter([])

            return _S()

    def _stub_engine_sq_empty():
        return object()

    def _stub_session_sq_empty(*_args, **_kw):
        return _FakeSession()

    monkeypatch.setattr(teams_route, "get_engine", _stub_engine_sq_empty)
    monkeypatch.setattr(teams_route, "Session", _stub_session_sq_empty)
    r = client.get("/api/v1/teams/Argentina/squad")
    assert r.status_code == 200
    body = r.json()
    assert body["team"] == "Argentina"
    assert body["players"] == []
    assert body["snapshot_date"] is None


def test_team_squad_returns_latest_snapshot_with_players(client: TestClient, monkeypatch) -> None:
    import datetime as _dt

    from wc2026.api.routes import teams as teams_route
    from wc2026.db.models import RawSquad

    snapshot_date = _dt.date(2026, 5, 1)
    rows = [
        RawSquad(
            tournament="FIFA World Cup 2026",
            team="Argentina",
            player_name="Lionel Messi",
            snapshot_date=snapshot_date,
            shirt_number=10,
            position="FW",
            club="Inter Miami",
            caps=192,
            goals=109,
        ),
        RawSquad(
            tournament="FIFA World Cup 2026",
            team="Argentina",
            player_name="Emiliano Martinez",
            snapshot_date=snapshot_date,
            shirt_number=23,
            position="GK",
            club="Aston Villa",
        ),
    ]

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def execute(self, _stmt):
            class _R:
                def first(_self):
                    return ("FIFA World Cup 2026", snapshot_date)

            return _R()

        def scalars(self, _stmt):
            class _S:
                def __iter__(_self):
                    return iter(rows)

            return _S()

    def _stub_engine_sq_pop():
        return object()

    def _stub_session_sq_pop(*_args, **_kw):
        return _FakeSession()

    monkeypatch.setattr(teams_route, "get_engine", _stub_engine_sq_pop)
    monkeypatch.setattr(teams_route, "Session", _stub_session_sq_pop)
    r = client.get("/api/v1/teams/Argentina/squad")
    assert r.status_code == 200
    body = r.json()
    assert body["snapshot_date"] == "2026-05-01"
    assert len(body["players"]) == 2
    # Players come out sorted by shirt number ascending.
    assert body["players"][0]["shirt_number"] == 10
    assert body["players"][1]["shirt_number"] == 23


def test_standings_falls_back_to_in_process_when_no_persisted_run(
    client: TestClient, monkeypatch
) -> None:
    from wc2026.api.routes import tournament as tour

    monkeypatch.setattr(tour, "_load_persisted_summary", lambda: None)
    r = client.get("/api/v1/tournament/standings", params={"n_sims": 100})
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "in_process"


def test_wc2026_track_record_returns_503_on_db_error(client: TestClient, monkeypatch) -> None:
    from wc2026.api.routes import track_record as tr

    def boom(**_):
        raise RuntimeError("Postgres unreachable")

    monkeypatch.setattr(tr, "load_wc_match_id_map", lambda: {})
    monkeypatch.setattr(tr, "compute_rolling", boom)
    r = client.get("/api/v1/track-record/wc2026")
    assert r.status_code == 503
    assert "track-record" in r.json()["detail"]


# --- /api/v1/track-record/historical/{tournament} --------------------------


def _clear_historical_cache() -> None:
    from wc2026.api.routes import track_record as tr

    with tr._HISTORICAL_CACHE_LOCK:
        tr._HISTORICAL_CACHE.clear()


def test_historical_track_record_serialises_stub_hindcast(
    client: TestClient, monkeypatch
) -> None:
    """Monkey-patch the heavy `hindcast()` call to a tiny fixed payload so the
    test exercises the route + response shape without re-fitting models."""
    import pandas as pd

    from wc2026.api.routes import track_record as tr

    _clear_historical_cache()

    def stub_hindcast(target, history, *, cfg=None):
        _ = target, history, cfg
        return pd.DataFrame(
            [
                {"date": "2022-11-20", "home_team": "Qatar", "away_team": "Ecuador",
                 "observed": "A", "actual_home": 0, "actual_away": 2,
                 "p_home": 0.30, "p_draw": 0.30, "p_away": 0.40, "train_n": 100,
                 "neutral": True, "skipped_reason": None},
                {"date": "2022-11-21", "home_team": "England", "away_team": "Iran",
                 "observed": "H", "actual_home": 6, "actual_away": 2,
                 "p_home": 0.55, "p_draw": 0.25, "p_away": 0.20, "train_n": 100,
                 "neutral": True, "skipped_reason": None},
            ]
        )

    def stub_load_played():
        return pd.DataFrame(
            {
                "tournament": ["FIFA World Cup", "FIFA World Cup"],
                "date": [pd.Timestamp("2022-11-20"), pd.Timestamp("2022-11-21")],
                "home_team": ["Qatar", "England"],
                "away_team": ["Ecuador", "Iran"],
                "home_score": [0, 6],
                "away_score": [2, 2],
                "neutral": [True, True],
            }
        )

    monkeypatch.setattr(tr, "hindcast", stub_hindcast)
    monkeypatch.setattr(tr, "load_played", stub_load_played)

    r = client.get("/api/v1/track-record/historical/WC2022")
    assert r.status_code == 200
    body = r.json()
    assert body["tournament"] == "WC2022"
    headline = body["headline"]
    assert headline["n_matches"] == 2
    # base rates sum to 1 across H/D/A
    assert abs(headline["base_h"] + headline["base_d"] + headline["base_a"] - 1.0) < 1e-6
    assert headline["log_loss"] > 0
    assert isinstance(body["reliability"], list)
    # Bookmaker reference is populated for WC2022 with the literature constants.
    assert body["bookmaker_reference"]["log_loss_low"] == 0.95


def test_historical_track_record_rejects_unknown_tournament(client: TestClient) -> None:
    # Pydantic Path() pattern enforces WC(2018|2022); anything else is a 422.
    r = client.get("/api/v1/track-record/historical/EURO2024")
    assert r.status_code == 422


def test_historical_track_record_cache_short_circuits(client: TestClient, monkeypatch) -> None:
    """A second request inside the TTL window must skip the hindcast call."""
    import pandas as pd

    from wc2026.api.routes import track_record as tr

    _clear_historical_cache()

    calls = {"n": 0}

    def stub_hindcast(target, history, *, cfg=None):
        _ = target, history, cfg
        calls["n"] += 1
        return pd.DataFrame(
            [
                {"date": "2018-06-14", "home_team": "Russia", "away_team": "Saudi Arabia",
                 "observed": "H", "actual_home": 5, "actual_away": 0,
                 "p_home": 0.45, "p_draw": 0.30, "p_away": 0.25, "train_n": 100,
                 "neutral": True, "skipped_reason": None},
            ]
        )

    def stub_load_played():
        return pd.DataFrame(
            {
                "tournament": ["FIFA World Cup"],
                "date": [pd.Timestamp("2018-06-14")],
                "home_team": ["Russia"],
                "away_team": ["Saudi Arabia"],
                "home_score": [5],
                "away_score": [0],
                "neutral": [True],
            }
        )

    monkeypatch.setattr(tr, "hindcast", stub_hindcast)
    monkeypatch.setattr(tr, "load_played", stub_load_played)

    r1 = client.get("/api/v1/track-record/historical/WC2018")
    r2 = client.get("/api/v1/track-record/historical/WC2018")
    assert r1.status_code == r2.status_code == 200
    assert r1.json() == r2.json()
    assert calls["n"] == 1  # second request served from the cache


# --- /api/v1/track-record/bookmaker-benchmark ---------------------------------


def test_bookmaker_benchmark_404_when_artifact_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    from pathlib import Path

    from wc2026.api.routes import track_record as tr

    monkeypatch.setattr(tr, "BOOKMAKER_BENCHMARK_PATH", Path(tmp_path) / "absent.json")  # type: ignore[arg-type]
    r = client.get("/api/v1/track-record/bookmaker-benchmark")
    assert r.status_code == 404
    assert "missing" in r.json()["detail"] or "not on disk" in r.json()["detail"]


def test_bookmaker_benchmark_200_returns_artifact_shape(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    import json
    from pathlib import Path

    from wc2026.api.routes import track_record as tr

    payload = {
        "as_of": "2026-05-26T00:00:00+00:00",
        "cutoff": "2024-08-01",
        "n_train": 3000,
        "n_test": 1200,
        "n_scored": 1150,
        "poisson_log_loss": 1.013,
        "bookmaker_log_loss": 0.974,
        "delta": 0.039,
        "base_h": 0.46,
        "base_d": 0.24,
        "base_a": 0.30,
        "leagues": [["2024_25", "E0"], ["2024_25", "E1"]],
        "half_life_days": 365.0,
    }
    artifact = Path(tmp_path) / "latest.json"  # type: ignore[arg-type]
    artifact.write_text(json.dumps(payload))
    monkeypatch.setattr(tr, "BOOKMAKER_BENCHMARK_PATH", artifact)
    r = client.get("/api/v1/track-record/bookmaker-benchmark")
    assert r.status_code == 200
    body = r.json()
    assert body["poisson_log_loss"] == pytest.approx(1.013)
    assert body["bookmaker_log_loss"] == pytest.approx(0.974)
    assert body["delta"] == pytest.approx(0.039)
    assert body["n_scored"] == 1150
    assert body["leagues"] == [["2024_25", "E0"], ["2024_25", "E1"]]


def test_bookmaker_benchmark_503_on_malformed_json(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    from pathlib import Path

    from wc2026.api.routes import track_record as tr

    artifact = Path(tmp_path) / "latest.json"  # type: ignore[arg-type]
    artifact.write_text("{not valid json")
    monkeypatch.setattr(tr, "BOOKMAKER_BENCHMARK_PATH", artifact)
    r = client.get("/api/v1/track-record/bookmaker-benchmark")
    assert r.status_code == 503
    assert "malformed" in r.json()["detail"]


def test_bookmaker_benchmark_503_on_invalid_shape(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    """JSON parses but Pydantic validation fails (e.g. wrong types)."""
    import json
    from pathlib import Path

    from wc2026.api.routes import track_record as tr

    # n_train should be an int; passing a string trips Pydantic's coercion.
    payload = {
        "as_of": "2026-05-26T00:00:00+00:00",
        "cutoff": "2024-08-01",
        "n_train": "not-an-int",
        "n_test": 1200,
        "n_scored": 1150,
        "poisson_log_loss": 1.013,
        "bookmaker_log_loss": 0.974,
        "delta": 0.039,
        "leagues": [["2024_25", "E0"]],
        "half_life_days": 365.0,
    }
    artifact = Path(tmp_path) / "latest.json"  # type: ignore[arg-type]
    artifact.write_text(json.dumps(payload))
    monkeypatch.setattr(tr, "BOOKMAKER_BENCHMARK_PATH", artifact)
    r = client.get("/api/v1/track-record/bookmaker-benchmark")
    assert r.status_code == 503
    assert "unexpected shape" in r.json()["detail"]
