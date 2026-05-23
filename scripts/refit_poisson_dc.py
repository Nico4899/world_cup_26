"""Refit the bivariate Poisson + Dixon-Coles model and persist the parameters.

Used by the daily scheduler job; can also be run manually:

    uv run python scripts/refit_poisson_dc.py
    uv run python scripts/refit_poisson_dc.py --ref-date 2026-06-15

Writes ``data/artifacts/poisson_dc/latest.npz`` (overwriting) plus a dated copy
``data/artifacts/poisson_dc/YYYY-MM-DD.npz`` for audit history.

Also re-fits the Elo-based shootout submodel against the freshest eloratings
snapshot and persists it to ``data/artifacts/shootout/latest.json`` (with a
dated copy). If the snapshot or historical-shootouts CSV are missing, the
shootout refit is skipped with a warning — the Poisson refit still succeeds.

The API picks up the latest artefacts on lifespan startup; restart the API
service to use a freshly-refitted model.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from wc2026.features.match_weights import combined_weight
from wc2026.ingest.eloratings_scraper import load_latest_snapshot
from wc2026.ingest.kaggle_intl import load_played
from wc2026.models.poisson_dc import PoissonDC
from wc2026.models.shootout import fit_shootout_model, load_historical_shootouts

logger = logging.getLogger(__name__)

DEFAULT_ARTEFACT_DIR = Path("data/artifacts/poisson_dc")
DEFAULT_SHOOTOUT_DIR = Path("data/artifacts/shootout")
DEFAULT_HALF_LIFE_DAYS = 3650.0
DEFAULT_HISTORY_YEARS = 10


def _refit_shootout(
    ref_date: pd.Timestamp,
    shootout_dir: Path = DEFAULT_SHOOTOUT_DIR,
) -> Path | None:
    """Best-effort: refit the Elo-based shootout model from the latest snapshot.

    Returns the written latest.json path, or None if inputs are missing / the
    fit could not converge. Never raises — the daily job should still succeed
    if eloratings.net was briefly unreachable.
    """
    try:
        elo = load_latest_snapshot()
        shootouts = load_historical_shootouts()
        model = fit_shootout_model(shootouts, elo)
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("shootout refit skipped: %s", exc)
        return None
    dated = shootout_dir / f"{ref_date:%Y-%m-%d}.json"
    latest = shootout_dir / "latest.json"
    model.save(dated)
    model.save(latest)
    print(f"shootout refit: slope={model.slope:+.6f} n_train={model.n_train} → {latest}")
    return latest


def refit_and_save(
    ref_date: pd.Timestamp,
    *,
    artefact_dir: Path = DEFAULT_ARTEFACT_DIR,
    shootout_dir: Path = DEFAULT_SHOOTOUT_DIR,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    history_years: int = DEFAULT_HISTORY_YEARS,
) -> Path:
    """Fit a PoissonDC on the last ``history_years`` and persist to artefact_dir.

    Also re-fits the shootout submodel and writes it to ``shootout_dir``
    (best-effort: skipped silently if Elo or shootouts inputs are missing).
    Returns the path of the written latest.npz.
    """
    cutoff = ref_date - pd.Timedelta(days=int(history_years * 365.25))
    df = load_played()
    df = df[df["date"] >= cutoff].reset_index(drop=True)
    weights = combined_weight(df, ref_date=ref_date, half_life_days=half_life_days)
    t0 = time.time()
    model = PoissonDC().fit(df, weights=weights)
    fit_seconds = time.time() - t0

    artefact_dir.mkdir(parents=True, exist_ok=True)
    dated_path = artefact_dir / f"{ref_date:%Y-%m-%d}.npz"
    latest_path = artefact_dir / "latest.npz"
    model.params_.save(dated_path)
    model.params_.save(latest_path)
    print(
        f"refit complete in {fit_seconds:.2f}s · "
        f"n_teams={len(model.params_.teams)} · "
        f"home_advantage={model.params_.home_advantage:.4f} · "
        f"rho={model.params_.rho:.4f} · "
        f"written to {latest_path} (+ {dated_path.name})"
    )
    _refit_shootout(ref_date, shootout_dir=shootout_dir)
    return latest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ref-date",
        default=datetime.now(UTC).date().isoformat(),
        help="Reference date for the fit (default: today UTC).",
    )
    parser.add_argument(
        "--artefact-dir",
        default=str(DEFAULT_ARTEFACT_DIR),
        help=f"Output directory (default: {DEFAULT_ARTEFACT_DIR}).",
    )
    parser.add_argument(
        "--half-life-days",
        type=float,
        default=DEFAULT_HALF_LIFE_DAYS,
        help=f"Time-decay half-life (default: {DEFAULT_HALF_LIFE_DAYS}).",
    )
    parser.add_argument(
        "--history-years",
        type=int,
        default=DEFAULT_HISTORY_YEARS,
        help=f"Training-window years (default: {DEFAULT_HISTORY_YEARS}).",
    )
    args = parser.parse_args()

    refit_and_save(
        ref_date=pd.Timestamp(args.ref_date),
        artefact_dir=Path(args.artefact_dir),
        half_life_days=args.half_life_days,
        history_years=args.history_years,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
