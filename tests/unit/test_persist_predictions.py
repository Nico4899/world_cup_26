"""Unit tests for scripts/persist_wc2026_predictions.py.

Uses an in-memory SQLite engine + the real fixture loader + a tiny PoissonDC
fit so the persisted rows match the contract callers downstream expect.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest
import scripts.persist_wc2026_predictions as p
from sqlalchemy import create_engine, select

from wc2026.db.models import Base, ModelPrediction
from wc2026.features.match_weights import combined_weight
from wc2026.ingest.kaggle_intl import load_played
from wc2026.models.poisson_dc import PoissonDC


@pytest.fixture
def sqlite_engine():
    eng = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(eng)
    try:
        yield eng
    finally:
        eng.dispose()


def _fitted_model() -> PoissonDC:
    """Cheap fit on the recent corpus to make scoring 72 fixtures fast in tests."""
    df = load_played()
    cutoff = pd.Timestamp("2020-01-01")
    train = df[df["date"] >= cutoff].reset_index(drop=True)
    weights = combined_weight(train, ref_date=pd.Timestamp("2025-01-01"), half_life_days=3650.0)
    return PoissonDC().fit(train, weights=weights)


def test_build_prediction_rows_emits_one_row_per_fixture() -> None:
    fixtures = p._load_fixtures()
    rows = p.build_prediction_rows(fixtures.matches, _fitted_model())
    assert len(rows) == 72
    assert {r["model_version"] for r in rows} == {"poisson_dc.v1"}


def test_build_prediction_rows_probabilities_sum_to_one() -> None:
    fixtures = p._load_fixtures()
    rows = p.build_prediction_rows(fixtures.matches, _fitted_model())
    for row in rows:
        s = row["p_home"] + row["p_draw"] + row["p_away"]
        assert abs(s - 1.0) < 1e-6


def test_build_prediction_rows_carries_score_matrix_by_default() -> None:
    fixtures = p._load_fixtures()
    rows = p.build_prediction_rows(fixtures.matches[:3], _fitted_model())
    for row in rows:
        matrix = row["score_matrix_json"]
        assert matrix is not None
        # 11x11 (max_goals=10), each row sums to a positive probability.
        assert len(matrix) == 11
        assert len(matrix[0]) == 11


def test_build_prediction_rows_skips_matrix_when_requested() -> None:
    fixtures = p._load_fixtures()
    rows = p.build_prediction_rows(fixtures.matches[:3], _fitted_model(), include_matrix=False)
    for row in rows:
        assert row["score_matrix_json"] is None


def test_build_prediction_rows_uses_uniform_fallback_for_unknown_team(monkeypatch) -> None:
    """If the model raises KeyError for a team, the row still gets a uniform 1/3 triplet."""
    from wc2026.sim.fixtures import FixtureMatch

    fixture = FixtureMatch(
        date=pd.Timestamp("2026-06-11"),
        home_team="Atlantis",  # not in the fitted set
        away_team="Atlantica",
        group="A",
        city="MetLife",
        country="USA",
        neutral=False,
    )
    rows = p.build_prediction_rows([fixture], _fitted_model())
    assert len(rows) == 1
    assert abs(rows[0]["p_home"] - 1 / 3) < 1e-9
    assert rows[0]["score_matrix_json"] is None


def test_persist_rows_writes_to_db(sqlite_engine) -> None:
    fixtures = p._load_fixtures()
    rows = p.build_prediction_rows(fixtures.matches[:5], _fitted_model())
    n = p.persist_rows(rows, engine=sqlite_engine)
    assert n == 5
    with sqlite_engine.connect() as conn:
        stored = list(conn.execute(select(ModelPrediction)).all())
    assert len(stored) == 5


def test_persist_rows_accumulates_daily_snapshots(sqlite_engine) -> None:
    """Two calls with different created_at append (no unique key on the natural fields)."""
    fixtures = p._load_fixtures()
    day1 = datetime(2026, 5, 23, tzinfo=UTC)
    day2 = datetime(2026, 5, 24, tzinfo=UTC)
    rows1 = p.build_prediction_rows(fixtures.matches[:3], _fitted_model(), now=day1)
    rows2 = p.build_prediction_rows(fixtures.matches[:3], _fitted_model(), now=day2)
    p.persist_rows(rows1, engine=sqlite_engine)
    p.persist_rows(rows2, engine=sqlite_engine)
    with sqlite_engine.connect() as conn:
        n = conn.execute(select(ModelPrediction)).all()
    assert len(n) == 6


def test_persist_daily_snapshot_no_op_without_artefact(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    n = p.persist_daily_snapshot(artefact_path=tmp_path / "missing.npz")
    assert n == 0


def test_persist_daily_snapshot_no_op_without_database_url(monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("WC2026_DATABASE_URL", raising=False)
    n = p.persist_daily_snapshot()
    assert n == 0
