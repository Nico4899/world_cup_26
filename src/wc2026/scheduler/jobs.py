"""APScheduler entry point: daily refresh jobs for ingest sources.

Schedule (UTC)
    04:00  Jürisoo Kaggle dataset refresh
    04:15  Elo ratings snapshot
    04:30  football-data.org WC fixtures refresh
    05:00  Poisson + Dixon-Coles model refit → data/artefacts/poisson_dc/latest.npz

Each job writes a row to ``scheduler_job_runs`` so the run history is visible
in Postgres. If the DB is unavailable, the failure is logged but does not
abort the scheduler process — the next tick will retry.

The model refit overwrites ``data/artefacts/poisson_dc/latest.npz``; the API
loads that on the next lifespan startup. Restart the API container (or the
``api`` service in docker-compose) to pick up a freshly-refitted model.

Run with:
    python -m wc2026.scheduler.jobs
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from wc2026.db.models import SchedulerJobRun
from wc2026.db.session import session_scope
from wc2026.ingest.eloratings_scraper import fetch_current_ratings
from wc2026.ingest.football_data_org import (
    WC_COMPETITION_CODE,
    fetch_competition_matches,
)
from wc2026.ingest.kaggle_intl import download_dataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JobSpec:
    name: str
    hour: int
    minute: int
    func: Callable[..., Any]


def _job_kaggle_refresh() -> None:
    download_dataset(force=True)


def _job_elo_refresh() -> None:
    fetch_current_ratings()


def _job_football_data_refresh() -> None:
    if not os.environ.get("FOOTBALL_DATA_ORG_KEY"):
        logger.warning("FOOTBALL_DATA_ORG_KEY not set — skipping WC fixtures refresh")
        return
    fetch_competition_matches(WC_COMPETITION_CODE)


def _job_poisson_refit() -> None:
    # Local import: keeps the scheduler module importable without scipy/sklearn
    # at parse time, and matches the lazy-import pattern used by ingesters.
    from scripts.refit_poisson_dc import refit_and_save  # noqa: PLC0415

    refit_and_save(ref_date=pd.Timestamp(datetime.now(UTC).date()))


JOB_SPECS: tuple[JobSpec, ...] = (
    JobSpec(name="kaggle_refresh", hour=4, minute=0, func=_job_kaggle_refresh),
    JobSpec(name="elo_refresh", hour=4, minute=15, func=_job_elo_refresh),
    JobSpec(
        name="football_data_org_refresh",
        hour=4,
        minute=30,
        func=_job_football_data_refresh,
    ),
    JobSpec(name="poisson_refit", hour=5, minute=0, func=_job_poisson_refit),
)


def _record_job_run(name: str, started_at: datetime, status: str, error_text: str | None) -> None:
    """Persist a SchedulerJobRun row; swallow DB errors so the scheduler keeps running."""
    try:
        with session_scope() as s:
            s.add(
                SchedulerJobRun(
                    job_name=name,
                    started_at=started_at,
                    finished_at=datetime.now(UTC),
                    status=status,
                    error_text=error_text,
                )
            )
    except Exception:
        logger.exception("Failed to record scheduler_job_runs row for %s", name)


def _wrap_with_tracking(spec: JobSpec) -> Callable[[], None]:
    # The returned closure captures `spec`, so it can only be used with
    # APScheduler's MemoryJobStore (the default). Switching to a persistent
    # store (SQLAlchemyJobStore etc.) would require a top-level function and
    # passing spec.name via the job's args kwarg instead.
    def runner() -> None:
        started = datetime.now(UTC)
        try:
            spec.func()
        except Exception as exc:
            logger.exception("Job %s failed", spec.name)
            _record_job_run(spec.name, started, "error", repr(exc))
            return
        _record_job_run(spec.name, started, "ok", None)

    runner.__name__ = f"{spec.name}_tracked"
    return runner


def register_jobs(
    scheduler: BlockingScheduler | BackgroundScheduler,
    specs: tuple[JobSpec, ...] = JOB_SPECS,
) -> None:
    """Attach every spec to the scheduler as a UTC cron trigger."""
    for spec in specs:
        scheduler.add_job(
            _wrap_with_tracking(spec),
            trigger=CronTrigger(hour=spec.hour, minute=spec.minute, timezone="UTC"),
            id=spec.name,
            name=spec.name,
            replace_existing=True,
        )


def build_scheduler() -> BlockingScheduler:
    scheduler = BlockingScheduler(timezone="UTC")
    register_jobs(scheduler)
    return scheduler


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    scheduler = build_scheduler()
    logger.info("Starting scheduler with %d jobs", len(JOB_SPECS))
    scheduler.start()


if __name__ == "__main__":
    main()


__all__ = [
    "JOB_SPECS",
    "JobSpec",
    "build_scheduler",
    "main",
    "register_jobs",
]
