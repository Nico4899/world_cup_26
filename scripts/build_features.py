"""Materialise the per-match feature table for WC 2026 fixtures.

Loads every Phase 2/3 source we have on disk, fits a per-day PoissonDC if no
artefact is present, computes features for every WC 2026 fixture, and upserts
into ``features_match_features``. Missing sources yield NaN features — the
script never aborts on a missing input.

CLI usage:
    uv run python scripts/build_features.py
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import insert as sa_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine

from wc2026.db.models import MatchFeatures
from wc2026.db.session import get_engine, session_scope
from wc2026.features.build_match_features import (
    FeatureSources,
    MatchSpec,
    build_features_for_matches,
)
from wc2026.ingest.eloratings_scraper import load_latest_snapshot as load_latest_elo
from wc2026.ingest.kaggle_intl import load_played, load_scheduled
from wc2026.ingest.statsbomb_open import (
    aggregate_match_xg,
    load_shots_corpus,
)
from wc2026.models.poisson_dc import PoissonDC, PoissonDCParams
from wc2026.sim.fixtures import parse_wc2026_fixtures

# Mirrors api.main.MODEL_VERSION; we duplicate the constant rather than import
# from the API module so this script stays usable without the FastAPI deps.
POISSON_MODEL_VERSION = "poisson_dc.v1"

DEFAULT_POISSON_ARTIFACT = Path("data/artifacts/poisson_dc/latest.npz")
DEFAULT_FIFA_DIR = Path("data/raw/wikipedia")
DEFAULT_SQUADS_DIR = Path("data/raw/wikipedia")

logger = logging.getLogger(__name__)


# ---- source loaders --------------------------------------------------------


def _load_elo_by_team() -> tuple[dict[str, float] | None, str | None]:
    """Return ({team_name: rating}, snapshot_date_iso) from the latest elo snapshot.

    Returns ``(None, None)`` if no snapshot is on disk.
    """
    try:
        df = load_latest_elo()
    except FileNotFoundError:
        return None, None
    if "team_name" not in df.columns or "rating" not in df.columns:
        return None, None
    by_team = {
        str(name): float(rating)
        for name, rating in zip(df["team_name"], df["rating"], strict=True)
        if pd.notna(name) and pd.notna(rating)
    }
    snapshot_date = None
    if "snapshot_date" in df.columns and not df["snapshot_date"].empty:
        snapshot_date = pd.Timestamp(df["snapshot_date"].iloc[0]).date().isoformat()
    return by_team, snapshot_date


def _load_fifa_rank_by_team(
    target_dir: Path = DEFAULT_FIFA_DIR,
) -> tuple[dict[str, int] | None, str | None]:
    paths = sorted(target_dir.glob("fifa_ranking_*.parquet"))
    if not paths:
        return None, None
    df = pd.read_parquet(paths[-1])
    if "team" not in df.columns or "rank" not in df.columns:
        return None, None
    by_team = {
        str(t): int(r)
        for t, r in zip(df["team"], df["rank"], strict=True)
        if pd.notna(t) and pd.notna(r)
    }
    snapshot_date = paths[-1].stem.removeprefix("fifa_ranking_")
    return by_team, snapshot_date


def _load_xg_history() -> pd.DataFrame | None:
    """Aggregate per-match per-team xG from the StatsBomb shots corpus on disk."""
    shots = load_shots_corpus()
    if shots.empty:
        return None
    agg = aggregate_match_xg(shots)
    if agg.empty:
        return None
    # build_match_features wants (match_date, team, xg_for, xg_against)
    return agg[["match_date", "team", "xg_for", "xg_against"]].copy()


def _load_squad_age_by_team(
    target_dir: Path = DEFAULT_SQUADS_DIR,
    *,
    ref_date: date | None = None,
) -> tuple[dict[str, float] | None, str | None]:
    paths = sorted(target_dir.glob("squads_*.parquet"))
    if not paths:
        return None, None
    df = pd.read_parquet(paths[-1])
    if "team" not in df.columns or "birth_date" not in df.columns:
        return None, None
    df = df.dropna(subset=["birth_date"])
    if df.empty:
        return None, None
    ref = pd.Timestamp(ref_date or datetime.now(UTC).date())
    df["age_years"] = (ref - pd.to_datetime(df["birth_date"])).dt.days / 365.25
    by_team = df.groupby("team")["age_years"].mean().to_dict()
    snapshot_date = paths[-1].stem
    return {str(k): float(v) for k, v in by_team.items()}, snapshot_date


def _load_poisson_model(
    artefact_path: Path = DEFAULT_POISSON_ARTIFACT,
) -> PoissonDC | None:
    """Rehydrate the ``.npz`` artefact into a PoissonDC ready for prediction.

    Mirrors ``api.main._load_or_fit_model`` so we don't duplicate the assembly
    logic. Returns ``None`` if the artefact is missing or unreadable.
    """
    if not artefact_path.exists():
        return None
    try:
        params = PoissonDCParams.load(artefact_path)
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("Could not load PoissonDC artefact %s: %s", artefact_path, exc)
        return None
    model = PoissonDC()
    model.params_ = params
    model._team_idx = {t: i for i, t in enumerate(params.teams)}
    model.converged_ = True
    return model


# ---- main entry-point ------------------------------------------------------


def assemble_sources(
    *,
    poisson_artefact: Path = DEFAULT_POISSON_ARTIFACT,
    ref_date: date | None = None,
) -> FeatureSources:
    """Pull every Phase 2/3 source off disk into a single FeatureSources bundle.

    Each loader returns ``None`` on absence; the bundle just rolls them up.
    """
    snapshot_meta: dict[str, Any] = {}

    elo, elo_date = _load_elo_by_team()
    if elo_date:
        snapshot_meta["elo_snapshot_date"] = elo_date

    fifa, fifa_date = _load_fifa_rank_by_team()
    if fifa_date:
        snapshot_meta["fifa_ranking_date"] = fifa_date

    xg_history = _load_xg_history()
    if xg_history is not None:
        snapshot_meta["xg_history_rows"] = len(xg_history)

    squads, squads_snapshot = _load_squad_age_by_team(ref_date=ref_date)
    if squads_snapshot:
        snapshot_meta["squads_snapshot"] = squads_snapshot

    poisson = _load_poisson_model(poisson_artefact)
    if poisson is not None:
        snapshot_meta["poisson_model_version"] = POISSON_MODEL_VERSION
        snapshot_meta["poisson_artefact"] = str(poisson_artefact)

    try:
        matches = load_played()
    except FileNotFoundError:
        matches = None

    return FeatureSources(
        elo_by_team=elo,
        fifa_rank_by_team=fifa,
        xg_history=xg_history,
        squad_age_by_team=squads,
        matches=matches,
        poisson_model=poisson,
        snapshot_meta=snapshot_meta,
    )


def _wc2026_match_specs() -> list[MatchSpec]:
    """Convert the 72 group-stage fixtures into MatchSpec objects."""
    scheduled = load_scheduled()
    fixtures = parse_wc2026_fixtures(scheduled)
    return [
        MatchSpec(
            match_date=m.date.date() if hasattr(m.date, "date") else m.date,
            home_team=m.home_team,
            away_team=m.away_team,
            neutral=m.neutral,
        )
        for m in fixtures.matches
    ]


_FEATURE_COLUMNS: frozenset[str] = frozenset(c.name for c in MatchFeatures.__table__.columns)


def _coerce_features_row(row: dict[str, Any]) -> dict[str, Any]:
    """Adapt the orchestrator's dict to the MatchFeatures column shape.

    Numeric ``None``/``NaN`` are normalised to ``None``; integer flags stay
    int. Keys that aren't yet columns on ``MatchFeatures`` (e.g. new
    venue/altitude features waiting on an Alembic migration) are silently
    dropped so the persistence layer doesn't break on schema lag.
    """
    coerced: dict[str, Any] = {}
    for k, v in row.items():
        if k not in _FEATURE_COLUMNS:
            continue
        if v is None:
            coerced[k] = None
            continue
        if isinstance(v, float) and np.isnan(v):
            coerced[k] = None
            continue
        coerced[k] = v
    coerced["built_at"] = datetime.now(UTC)
    return coerced


def _upsert_rows(engine: Engine, rows: list[dict[str, Any]]) -> int:
    """Upsert ``rows`` into ``features_match_features`` (Postgres / SQLite)."""
    if not rows:
        return 0
    dialect = engine.dialect.name
    update_cols = {
        col.name
        for col in MatchFeatures.__table__.columns
        if col.name not in {"match_date", "home_team", "away_team"}
    }
    with engine.begin() as conn:
        if dialect == "postgresql":
            stmt = pg_insert(MatchFeatures.__table__).values(rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=["match_date", "home_team", "away_team"],
                set_={c: stmt.excluded[c] for c in update_cols},
            )
        elif dialect == "sqlite":
            stmt = sqlite_insert(MatchFeatures.__table__).values(rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=["match_date", "home_team", "away_team"],
                set_={c: stmt.excluded[c] for c in update_cols},
            )
        else:
            # Best-effort fallback: plain insert (will fail on duplicate PK).
            stmt = sa_insert(MatchFeatures.__table__).values(rows)
        conn.execute(stmt)
    return len(rows)


def build_and_persist_features(
    *,
    ref_date: date | None = None,
    engine: Engine | None = None,
    poisson_artefact: Path = DEFAULT_POISSON_ARTIFACT,
) -> int:
    """Top-level entrypoint: assemble sources, build rows for WC 2026, upsert.

    Returns the number of rows written. Designed to be called by the scheduler
    immediately after ``poisson_refit`` so the table reflects the latest model.
    """
    sources = assemble_sources(poisson_artefact=poisson_artefact, ref_date=ref_date)
    specs = _wc2026_match_specs()
    if not specs:
        logger.warning("build_features: no WC 2026 fixtures available — skipping")
        return 0
    df = build_features_for_matches(specs, sources)
    rows = [_coerce_features_row(r) for r in df.to_dict(orient="records")]
    eng = engine or get_engine()
    n = _upsert_rows(eng, rows)
    logger.info("build_features: upserted %d feature rows", n)
    return n


# ---- minimal session-aware wrappers for tests ------------------------------


def read_persisted_count(engine: Engine | None = None) -> int:
    """Helper for tests: return the row count in features_match_features."""
    eng = engine or get_engine()
    with session_scope(eng.url.render_as_string(hide_password=False)) as s:
        return s.query(MatchFeatures).count()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    n = build_and_persist_features()
    logger.info("Done. %d rows materialised.", n)


if __name__ == "__main__":
    main()
