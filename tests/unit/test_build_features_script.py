"""End-to-end test for scripts/build_features.py against an in-memory SQLite.

Verifies the upsert path, idempotency, and that missing upstream sources
yield NaN features without aborting the run.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import scripts.build_features as bf
from sqlalchemy import create_engine, select

from wc2026.db.models import Base, MatchFeatures
from wc2026.features.build_match_features import (
    FeatureSources,
    MatchSpec,
    build_features_for_matches,
)


@pytest.fixture
def sqlite_engine():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


def _two_specs() -> list[MatchSpec]:
    return [
        MatchSpec(date(2026, 6, 11), "Mexico", "Senegal", neutral=False),
        MatchSpec(date(2026, 6, 11), "Morocco", "Portugal", neutral=True),
    ]


def test_upsert_writes_rows_into_features_table(sqlite_engine) -> None:
    df = build_features_for_matches(_two_specs(), FeatureSources())
    rows = [bf._coerce_features_row(r) for r in df.to_dict(orient="records")]
    n = bf._upsert_rows(sqlite_engine, rows)
    assert n == 2
    with sqlite_engine.connect() as conn:
        stored = conn.execute(select(MatchFeatures)).all()
    assert len(stored) == 2


def test_upsert_is_idempotent_on_same_pk(sqlite_engine) -> None:
    """Running upsert twice with identical specs leaves exactly two rows, not four."""
    df = build_features_for_matches(_two_specs(), FeatureSources())
    rows = [bf._coerce_features_row(r) for r in df.to_dict(orient="records")]
    bf._upsert_rows(sqlite_engine, rows)
    bf._upsert_rows(sqlite_engine, rows)
    with sqlite_engine.connect() as conn:
        count = conn.execute(select(MatchFeatures)).all()
    assert len(count) == 2


def test_upsert_updates_feature_value_on_pk_collision(sqlite_engine) -> None:
    """Second upsert should *update* feature values, not insert duplicates."""
    df = build_features_for_matches(_two_specs(), FeatureSources())
    rows = [bf._coerce_features_row(r) for r in df.to_dict(orient="records")]
    bf._upsert_rows(sqlite_engine, rows)
    # Bump elo_diff for the first row.
    rows[0]["elo_diff"] = 999.0
    bf._upsert_rows(sqlite_engine, rows)
    with sqlite_engine.connect() as conn:
        stored = conn.execute(
            select(MatchFeatures.elo_diff).where(
                (MatchFeatures.home_team == "Mexico") & (MatchFeatures.away_team == "Senegal")
            )
        ).scalar_one()
    assert stored == 999.0


def test_upsert_no_op_on_empty_input(sqlite_engine) -> None:
    n = bf._upsert_rows(sqlite_engine, [])
    assert n == 0


def test_coerce_features_row_drops_nan_floats() -> None:
    raw = {
        "match_date": date(2026, 6, 11),
        "home_team": "X",
        "away_team": "Y",
        "elo_diff": float("nan"),
        "fifa_rank_diff": None,
        "rest_days_diff": 3.0,
        "is_neutral": 0,
        "is_host_home": 0,
        "is_host_away": 0,
    }
    out = bf._coerce_features_row(raw)
    assert out["elo_diff"] is None
    assert out["fifa_rank_diff"] is None
    assert out["rest_days_diff"] == 3.0
    assert "built_at" in out


def test_load_xg_history_returns_none_when_corpus_missing(monkeypatch, tmp_path: Path) -> None:
    """The loader must not crash when there's no on-disk StatsBomb corpus."""

    def _empty_corpus() -> pd.DataFrame:
        return pd.DataFrame()

    monkeypatch.setattr(bf, "load_shots_corpus", _empty_corpus)
    assert bf._load_xg_history() is None


def test_assemble_sources_returns_empty_bundle_when_nothing_on_disk(
    monkeypatch, tmp_path: Path
) -> None:
    """Empty disk → empty bundle, no exceptions."""
    monkeypatch.setattr(bf, "_load_elo_by_team", lambda: (None, None))
    monkeypatch.setattr(bf, "_load_fifa_rank_by_team", lambda *_, **__: (None, None))
    monkeypatch.setattr(bf, "_load_xg_history", lambda: None)
    monkeypatch.setattr(bf, "_load_squad_age_by_team", lambda *_, **__: (None, None))
    monkeypatch.setattr(bf, "_load_poisson_model", lambda *_, **__: None)
    monkeypatch.setattr(
        bf, "load_played", lambda *_, **__: (_ for _ in ()).throw(FileNotFoundError)
    )

    sources = bf.assemble_sources(poisson_artefact=tmp_path / "missing.npz")
    assert sources.elo_by_team is None
    assert sources.fifa_rank_by_team is None
    assert sources.xg_history is None
    assert sources.squad_age_by_team is None
    assert sources.poisson_model is None
    assert sources.matches is None
    assert sources.snapshot_meta == {}


def test_build_and_persist_features_writes_72_rows_when_fixtures_resolve(
    monkeypatch, sqlite_engine
) -> None:
    """Smoke test: when fixtures resolve and sources are empty, we still emit
    72 rows (the WC 2026 group-stage match count) with NaN feature columns."""
    # Use the empty-sources path so we don't depend on actual data on disk.
    monkeypatch.setattr(
        bf,
        "assemble_sources",
        lambda *_, **__: FeatureSources(snapshot_meta={"test": True}),
    )
    n = bf.build_and_persist_features(engine=sqlite_engine)
    assert n == 72
    with sqlite_engine.connect() as conn:
        rows = conn.execute(select(MatchFeatures)).all()
    assert len(rows) == 72
    # Every row should have host flags computed (independent of upstream data).
    with sqlite_engine.connect() as conn:
        host_home_sum = conn.execute(select(MatchFeatures.is_host_home)).scalars().all()
    # Three host teams play three group-stage matches each → 9 rows where home is host.
    assert sum(host_home_sum) == 9


def test_build_and_persist_returns_zero_when_no_fixtures(monkeypatch, sqlite_engine) -> None:
    monkeypatch.setattr(bf, "_wc2026_match_specs", lambda: [])
    n = bf.build_and_persist_features(engine=sqlite_engine)
    assert n == 0
