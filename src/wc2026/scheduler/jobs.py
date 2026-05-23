"""APScheduler entry point: daily refresh jobs for ingest sources.

Schedule (UTC)
    02:00  Postgres backup → data/backups/wc2026-YYYY-MM-DD.sql.gz (14-day retention)
    04:00  Jürisoo Kaggle dataset refresh
    04:15  Elo ratings snapshot
    04:30  football-data.org WC fixtures refresh
    05:00  Poisson + Dixon-Coles model refit → data/artifacts/poisson_dc/latest.npz
           (also refits the Elo-based shootout submodel → data/artifacts/shootout/latest.json)

Tournament-only (registered only when today ∈ [2026-06-11, 2026-07-19]):
    every 60 min  Warm the API's /tournament/standings cache via HTTP so the
                  dashboard sees a freshly-recomputed Monte Carlo on the next
                  refresh. Does NOT condition on completed matches yet — the
                  simulator still sims every match from scratch using the latest
                  daily-refit model parameters. Conditional re-simulation
                  against in-tournament results is a larger feature deferred
                  until a results-ingest pipeline lands.

Each job writes a row to ``scheduler_job_runs`` so the run history is visible
in Postgres. If the DB is unavailable, the failure is logged but does not
abort the scheduler process — the next tick will retry.

The model refit overwrites ``data/artifacts/poisson_dc/latest.npz``; the API
loads that on the next lifespan startup. Restart the API container (or the
``api`` service in docker-compose) to pick up a freshly-refitted model.

Run with:
    python -m wc2026.scheduler.jobs
"""

from __future__ import annotations

import gzip
import logging
import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from wc2026.db.models import SchedulerJobRun
from wc2026.db.session import session_scope
from wc2026.ingest.eloratings_scraper import fetch_current_ratings
from wc2026.ingest.football_data_org import (
    WC_COMPETITION_CODE,
    fetch_competition_matches,
)
from wc2026.ingest.kaggle_intl import download_dataset

DEFAULT_BACKUP_DIR = Path("data/backups")
BACKUP_RETENTION_DAYS = 14

WC_TOURNAMENT_START = date(2026, 6, 11)
WC_TOURNAMENT_END = date(2026, 7, 19)
STANDINGS_WARM_INTERVAL_MINUTES = 60

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


def _prune_backups(backup_dir: Path, retention_days: int = BACKUP_RETENTION_DAYS) -> int:
    """Delete .sql.gz backups older than ``retention_days``. Returns count removed."""
    if not backup_dir.exists():
        return 0
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    removed = 0
    for f in backup_dir.glob("wc2026-*.sql.gz"):
        mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=UTC)
        if mtime < cutoff:
            f.unlink()
            removed += 1
    return removed


def _job_db_backup(
    backup_dir: Path = DEFAULT_BACKUP_DIR,
    retention_days: int = BACKUP_RETENTION_DAYS,
) -> Path | None:
    """Dump DATABASE_URL to a gzipped .sql file; prune older than retention.

    Returns the written path, or None if skipped (no DATABASE_URL, pg_dump missing).
    Errors during dump propagate so the job-runs row records the failure.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        logger.warning("DATABASE_URL not set — skipping DB backup")
        return None
    if shutil.which("pg_dump") is None:
        logger.warning("pg_dump not on PATH — skipping DB backup (install postgresql-client)")
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(UTC).date()
    out_path = backup_dir / f"wc2026-{today.isoformat()}.sql.gz"
    with gzip.open(out_path, "wb") as fout:
        proc = subprocess.run(
            ["pg_dump", "--no-owner", "--no-privileges", db_url],
            capture_output=True,
            check=True,
        )
        fout.write(proc.stdout)
    removed = _prune_backups(backup_dir, retention_days)
    logger.info("DB backup written to %s (pruned %d old)", out_path, removed)
    return out_path


def _job_poisson_refit() -> None:
    # Local import: keeps the scheduler module importable without scipy/sklearn
    # at parse time, and matches the lazy-import pattern used by ingesters.
    from scripts.refit_poisson_dc import refit_and_save  # noqa: PLC0415

    refit_and_save(ref_date=pd.Timestamp(datetime.now(UTC).date()))


def _job_warm_standings_cache(api_url: str | None = None, timeout_s: float = 60.0) -> None:
    """Force the API to recompute its standings cache against the latest model.

    No-op outside the tournament window — but the job is also only *registered*
    inside the window, so this guard is belt-and-braces. Failures are logged but
    swallowed so a transient API outage doesn't fail the scheduler.
    """
    today = datetime.now(UTC).date()
    if not (WC_TOURNAMENT_START <= today <= WC_TOURNAMENT_END):
        return
    url = (api_url or os.environ.get("WC2026_API_URL", "http://localhost:8000")).rstrip("/")
    try:
        # Default n_sims; the API will recompute if its cache entry is older than its TTL.
        r = httpx.get(f"{url}/api/v1/tournament/standings", timeout=timeout_s)
        r.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("standings cache-warm hit %s failed: %s", url, exc)


JOB_SPECS: tuple[JobSpec, ...] = (
    JobSpec(name="db_backup", hour=2, minute=0, func=_job_db_backup),
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


def is_tournament_window(today: date | None = None) -> bool:
    """True when ``today`` (or now if None) is within the WC 2026 match window."""
    d = today or datetime.now(UTC).date()
    return WC_TOURNAMENT_START <= d <= WC_TOURNAMENT_END


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
    *,
    today: date | None = None,
) -> None:
    """Attach every cron spec to the scheduler.

    Inside the tournament window, also attach the interval-triggered standings
    cache-warm job. The ``today`` arg lets tests pin the window decision.
    """
    for spec in specs:
        scheduler.add_job(
            _wrap_with_tracking(spec),
            trigger=CronTrigger(hour=spec.hour, minute=spec.minute, timezone="UTC"),
            id=spec.name,
            name=spec.name,
            replace_existing=True,
        )
    if is_tournament_window(today):
        warm_spec = JobSpec(
            name="standings_cache_warm",
            hour=-1,
            minute=-1,
            func=_job_warm_standings_cache,
        )
        scheduler.add_job(
            _wrap_with_tracking(warm_spec),
            trigger=IntervalTrigger(minutes=STANDINGS_WARM_INTERVAL_MINUTES, timezone="UTC"),
            id=warm_spec.name,
            name=warm_spec.name,
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
    "STANDINGS_WARM_INTERVAL_MINUTES",
    "WC_TOURNAMENT_END",
    "WC_TOURNAMENT_START",
    "JobSpec",
    "build_scheduler",
    "is_tournament_window",
    "main",
    "register_jobs",
]
