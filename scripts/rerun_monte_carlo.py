"""Re-run the 10k tournament Monte Carlo conditioned on completed matches.

Pipeline:

1. Load the latest PoissonDC artefact + (optionally) the shootout submodel.
2. Read every FT_WHISTLE row in ``raw_live_events`` and look it up against the
   cached football-data.org WC 2026 fixtures to build a
   ``{(home, away): (h_score, a_score)}`` map.
3. Run ``simulate_tournament_monte_carlo(n_sims=10_000, ..., known_group_results=...)``.
4. Persist the run to ``tournament_sim_runs`` + ``tournament_sim_team_outcomes``.

The Phase 8 standings endpoint reads the most-recent persisted run rather
than re-running the simulator on every request. When the live event poller
writes new FT_WHISTLE rows, this script is rerun (manually or via the
scheduler) and the next standings request picks up the conditioned numbers.

CLI usage::

    uv run python scripts/rerun_monte_carlo.py
    uv run python scripts/rerun_monte_carlo.py --n-sims=5000 --seed=7
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from wc2026.db.models import TournamentSimRun, TournamentSimTeamOutcome
from wc2026.db.session import get_engine
from wc2026.ingest.kaggle_intl import load_scheduled
from wc2026.models.poisson_dc import PoissonDC, PoissonDCParams
from wc2026.sim.conditional import known_group_results_from_live_events
from wc2026.sim.fixtures import load_group_assignment, parse_wc2026_fixtures
from wc2026.sim.tournament import simulate_tournament_monte_carlo

DEFAULT_ARTEFACT_PATH = Path("data/artifacts/poisson_dc/latest.npz")
DEFAULT_GROUP_ASSIGNMENT_PATH = Path("data/wc2026_group_assignment.json")
DEFAULT_MODEL_VERSION = "poisson_dc.v1"

logger = logging.getLogger(__name__)


def _hydrate_model(artefact_path: Path) -> PoissonDC:
    params = PoissonDCParams.load(artefact_path)
    model = PoissonDC()
    model.params_ = params
    model._team_idx = {t: i for i, t in enumerate(params.teams)}
    model.converged_ = True
    return model


def _load_fixtures():
    override = None
    if DEFAULT_GROUP_ASSIGNMENT_PATH.exists():
        try:
            override = load_group_assignment(DEFAULT_GROUP_ASSIGNMENT_PATH)
        except (OSError, ValueError):
            override = None
    return parse_wc2026_fixtures(load_scheduled(), override_assignment=override)


def _load_match_id_map() -> dict:
    """Best-effort: pull the football-data.org fixture cache to map match_id → fixture.

    Returns an empty dict when there's no cache or no API key — the conditional
    rerun then degenerates to an unconditional one (still useful: persists a
    fresh snapshot of standings).
    """
    try:
        from wc2026.ingest.football_data_org import (  # noqa: PLC0415
            WC_COMPETITION_CODE,
            fetch_competition_matches,
        )

        df = fetch_competition_matches(WC_COMPETITION_CODE)
    except Exception:
        logger.debug("football-data.org cache unavailable for MC rerun mapping", exc_info=True)
        return {}
    out = {}
    if df.empty:
        return out
    from datetime import date as _date  # noqa: PLC0415

    for _, row in df.iterrows():
        match_id = row.get("match_id")
        utc_date = row.get("utc_date")
        home = row.get("home_team")
        away = row.get("away_team")
        if match_id is None or utc_date is None or home is None or away is None:
            continue
        try:
            d = (
                utc_date.date()
                if hasattr(utc_date, "date")
                else _date.fromisoformat(str(utc_date)[:10])
            )
            out[int(match_id)] = (d, str(home), str(away))
        except (TypeError, ValueError):
            continue
    return out


def persist_run(
    *,
    engine: Engine,
    summary,
    model_version: str,
) -> int:
    """Insert one ``tournament_sim_runs`` row + 48 ``tournament_sim_team_outcomes``."""
    with Session(engine, future=True) as session:
        run = TournamentSimRun(
            created_at=datetime.now(UTC),
            n_sims=int(summary.n_sims),
            model_version=model_version,
        )
        session.add(run)
        session.flush()  # gets run.run_id
        for team in summary.probabilities.index:
            row = summary.probabilities.loc[team]
            session.add(
                TournamentSimTeamOutcome(
                    run_id=run.run_id,
                    team=team,
                    group_winner_p=float(row["group_winner"]),
                    group_runner_up_p=float(row["runner_up"]),
                    advance_r32_p=float(row["r32_reached"]),
                    advance_r16_p=float(row["r16_reached"]),
                    quarterfinal_p=float(row["qf_reached"]),
                    semifinal_p=float(row["sf_reached"]),
                    final_p=float(row["final_reached"]),
                    champion_p=float(row["champion"]),
                )
            )
        session.commit()
        return int(run.run_id)


def rerun_and_persist(
    *,
    n_sims: int = 10_000,
    seed: int = 42,
    artefact_path: Path = DEFAULT_ARTEFACT_PATH,
    model_version: str = DEFAULT_MODEL_VERSION,
    engine: Engine | None = None,
) -> int | None:
    """Top-level entrypoint. Returns the new ``run_id`` (or None when skipped)."""
    if engine is None and not (
        os.environ.get("DATABASE_URL") or os.environ.get("WC2026_DATABASE_URL")
    ):
        logger.warning("neither DATABASE_URL nor WC2026_DATABASE_URL set — skipping MC rerun")
        return None
    if not artefact_path.exists():
        logger.warning("no PoissonDC artefact at %s — run refit_poisson_dc first", artefact_path)
        return None
    model = _hydrate_model(artefact_path)
    fixtures = _load_fixtures()
    mapping = _load_match_id_map()
    eng = engine or get_engine()
    known = known_group_results_from_live_events(mapping, engine=eng)
    summary = simulate_tournament_monte_carlo(
        fixtures,
        model,
        n_sims=n_sims,
        seed=seed,
        known_group_results=known,
    )
    run_id = persist_run(engine=eng, summary=summary, model_version=model_version)
    logger.info(
        "MC rerun persisted: run_id=%d  n_sims=%d  known_results=%d",
        run_id,
        n_sims,
        len(known),
    )
    return run_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sims", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artefact", default=str(DEFAULT_ARTEFACT_PATH))
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    rerun_and_persist(
        n_sims=args.n_sims,
        seed=args.seed,
        artefact_path=Path(args.artefact),
        model_version=args.model_version,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["persist_run", "rerun_and_persist"]
