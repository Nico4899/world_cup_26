"""Refit the bivariate Poisson + Dixon-Coles model and persist the parameters.

Used by the daily scheduler job; can also be run manually:

    uv run python scripts/refit_poisson_dc.py
    uv run python scripts/refit_poisson_dc.py --ref-date 2026-06-15

Writes ``data/artefacts/poisson_dc/latest.npz`` (overwriting) plus a dated copy
``data/artefacts/poisson_dc/YYYY-MM-DD.npz`` for audit history.

The API picks up the latest artefact on lifespan startup; restart the API
service to use a freshly-refitted model.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from wc2026.features.match_weights import combined_weight
from wc2026.ingest.kaggle_intl import load_played
from wc2026.models.poisson_dc import PoissonDC

DEFAULT_ARTEFACT_DIR = Path("data/artefacts/poisson_dc")
DEFAULT_HALF_LIFE_DAYS = 3650.0
DEFAULT_HISTORY_YEARS = 10


def refit_and_save(
    ref_date: pd.Timestamp,
    *,
    artefact_dir: Path = DEFAULT_ARTEFACT_DIR,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    history_years: int = DEFAULT_HISTORY_YEARS,
) -> Path:
    """Fit a PoissonDC on the last ``history_years`` and persist to artefact_dir.

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
