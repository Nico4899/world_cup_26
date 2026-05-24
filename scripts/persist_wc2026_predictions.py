"""Persist a daily snapshot of WC 2026 predictions to ``model_predictions``.

Runs after ``refit_poisson_dc`` so the persisted probabilities reflect the
freshest fit. For each of the 72 WC 2026 group-stage fixtures we write one
row:

    (match_date, home_team, away_team, p_home, p_draw, p_away,
     score_matrix_json, model_version, created_at)

The table has no natural-key unique constraint by design — each daily run
appends, giving us a per-day audit trail. The Phase 7 rolling-calibration
job picks the latest row per fixture made strictly *before* the match date.

CLI usage::

    uv run python scripts/persist_wc2026_predictions.py
    uv run python scripts/persist_wc2026_predictions.py --model-version=poisson_dc.v1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from wc2026.db.models import ModelPrediction
from wc2026.db.session import get_engine
from wc2026.ingest.kaggle_intl import load_scheduled
from wc2026.models.poisson_dc import PoissonDC, PoissonDCParams
from wc2026.sim.fixtures import FixtureMatch, load_group_assignment, parse_wc2026_fixtures

DEFAULT_ARTEFACT_PATH = Path("data/artifacts/poisson_dc/latest.npz")
DEFAULT_GROUP_ASSIGNMENT_PATH = Path("data/wc2026_group_assignment.json")
DEFAULT_MODEL_VERSION = "poisson_dc.v1"

logger = logging.getLogger(__name__)


def _hydrate_model(artefact_path: Path) -> PoissonDC:
    """Build a ``PoissonDC`` from a saved ``.npz`` artefact."""
    params = PoissonDCParams.load(artefact_path)
    model = PoissonDC()
    model.params_ = params
    model._team_idx = {t: i for i, t in enumerate(params.teams)}
    model.converged_ = True
    return model


def _load_fixtures():
    """Load the 72 scheduled WC 2026 fixtures, honoring an official override if present."""
    override = None
    if DEFAULT_GROUP_ASSIGNMENT_PATH.exists():
        try:
            override = load_group_assignment(DEFAULT_GROUP_ASSIGNMENT_PATH)
        except (OSError, ValueError):
            override = None
    return parse_wc2026_fixtures(load_scheduled(), override_assignment=override)


def build_prediction_rows(
    fixtures: Iterable[FixtureMatch],
    model: PoissonDC,
    *,
    model_version: str = DEFAULT_MODEL_VERSION,
    now: datetime | None = None,
    include_matrix: bool = True,
) -> list[dict]:
    """Score every fixture against ``model`` and return ``model_predictions`` rows."""
    created_at = now or datetime.now(UTC)
    rows: list[dict] = []
    for m in fixtures:
        match_date = m.date.date() if hasattr(m.date, "date") else m.date
        try:
            outcome = model.outcome_probs(m.home_team, m.away_team, neutral=m.neutral)
        except (KeyError, ValueError):
            # Tiny association not in the fitted set — keep going, just emit a
            # uniform fallback row so the snapshot covers all 72 fixtures.
            outcome = {"home_win": 1 / 3, "draw": 1 / 3, "away_win": 1 / 3}
            score_matrix = None
        else:
            score_matrix = (
                model.score_probs(m.home_team, m.away_team, neutral=m.neutral).tolist()
                if include_matrix
                else None
            )
        rows.append(
            {
                "match_date": match_date,
                "home_team": m.home_team,
                "away_team": m.away_team,
                "p_home": float(outcome["home_win"]),
                "p_draw": float(outcome["draw"]),
                "p_away": float(outcome["away_win"]),
                "score_matrix_json": score_matrix,
                "model_version": model_version,
                "created_at": created_at,
            }
        )
    return rows


def persist_rows(rows: list[dict], engine: Engine | None = None) -> int:
    """Bulk insert ``rows`` into ``model_predictions``. Returns # rows written.

    Uses the provided engine directly so tests can drive an in-memory SQLite
    fixture without round-tripping through ``session_scope`` (which rebuilds
    the engine and would lose the ``sqlite:///:memory:`` connection).
    """
    if not rows:
        return 0
    eng = engine or get_engine()
    with Session(eng, future=True) as session:
        try:
            session.add_all(ModelPrediction(**r) for r in rows)
            session.commit()
        except Exception:
            session.rollback()
            raise
    return len(rows)


def persist_daily_snapshot(
    *,
    artefact_path: Path = DEFAULT_ARTEFACT_PATH,
    model_version: str = DEFAULT_MODEL_VERSION,
    engine: Engine | None = None,
    now: datetime | None = None,
) -> int:
    """Top-level entrypoint: load fixtures + model, score, persist. Returns # rows.

    Skipped (returns 0) when there's no model artefact on disk or no
    ``DATABASE_URL`` / ``WC2026_DATABASE_URL`` env var is set. Mirrors the
    ``features_rebuild`` scheduler job's pre-flight checks.
    """
    if engine is None and not (
        os.environ.get("DATABASE_URL") or os.environ.get("WC2026_DATABASE_URL")
    ):
        logger.warning(
            "neither DATABASE_URL nor WC2026_DATABASE_URL set — skipping prediction snapshot"
        )
        return 0
    if not artefact_path.exists():
        logger.warning(
            "no PoissonDC artefact at %s — run refit_poisson_dc first", artefact_path
        )
        return 0
    model = _hydrate_model(artefact_path)
    fixtures = _load_fixtures()
    rows = build_prediction_rows(
        fixtures.matches, model, model_version=model_version, now=now
    )
    eng = engine or get_engine()
    n = persist_rows(rows, engine=eng)
    logger.info("persisted %d WC 2026 prediction rows (model_version=%s)", n, model_version)
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artefact",
        default=str(DEFAULT_ARTEFACT_PATH),
        help=f"PoissonDC artefact path (default: {DEFAULT_ARTEFACT_PATH}).",
    )
    parser.add_argument(
        "--model-version",
        default=DEFAULT_MODEL_VERSION,
        help=f"Model-version tag for the persisted rows (default: {DEFAULT_MODEL_VERSION}).",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    persist_daily_snapshot(
        artefact_path=Path(args.artefact),
        model_version=args.model_version,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "build_prediction_rows",
    "persist_daily_snapshot",
    "persist_rows",
]
