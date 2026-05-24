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
from wc2026.ingest.fbref import fetch_team_match_logs
from wc2026.ingest.football_data_co_uk import fetch_calibration_corpus
from wc2026.ingest.football_data_org import (
    WC_COMPETITION_CODE,
    fetch_competition_matches,
)
from wc2026.ingest.kaggle_intl import download_dataset
from wc2026.ingest.openfootball import fetch_cup_txt
from wc2026.ingest.statsbomb_open import fetch_all_tournament_shots
from wc2026.ingest.thesportsdb import fetch_team_assets
from wc2026.ingest.wikipedia import fetch_all_squads, fetch_fifa_ranking

DEFAULT_BACKUP_DIR = Path("data/backups")
BACKUP_RETENTION_DAYS = 14

WC_TOURNAMENT_START = date(2026, 6, 11)
WC_TOURNAMENT_END = date(2026, 7, 19)
STANDINGS_WARM_INTERVAL_MINUTES = 60
MONTE_CARLO_RERUN_INTERVAL_MINUTES = 30

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JobSpec:
    """Cron-triggered job.

    ``hour``/``minute`` are always set. ``day_of_week`` and ``day`` are optional
    APScheduler CronTrigger fields:

    * ``day_of_week="sun"`` → weekly on Sundays
    * ``day=1``             → monthly on the 1st
    * both unset            → daily

    Manual-only jobs (the squad ingest) use a sentinel ``hour=-1`` and are
    listed in :data:`MANUAL_ONLY_JOB_SPECS` rather than :data:`JOB_SPECS`. They
    can be invoked via ``/api/v1/_ops/run-job/{name}``.
    """

    name: str
    hour: int
    minute: int
    func: Callable[..., Any]
    day_of_week: str | None = None
    day: int | str | None = None


def _job_kaggle_refresh() -> None:
    download_dataset(force=True)


def _job_elo_refresh() -> None:
    fetch_current_ratings()


def _job_football_data_refresh() -> None:
    if not os.environ.get("FOOTBALL_DATA_ORG_KEY"):
        logger.warning("FOOTBALL_DATA_ORG_KEY not set — skipping WC fixtures refresh")
        return
    fetch_competition_matches(WC_COMPETITION_CODE)


def _job_thesportsdb_refresh() -> None:
    """Refresh crest/kit/stadium metadata for the WC 2026 teams.

    Resolves the team list from the latest Jürisoo fixtures so we always
    cover the 48 qualified teams without hard-coding them.
    """
    from wc2026.ingest.kaggle_intl import load_scheduled  # noqa: PLC0415
    from wc2026.sim.fixtures import parse_wc2026_fixtures  # noqa: PLC0415

    try:
        scheduled = load_scheduled()
        fixtures = parse_wc2026_fixtures(scheduled)
        team_names = list(fixtures.teams)
    except Exception:
        logger.exception("TheSportsDB refresh: could not resolve WC 2026 team list")
        return
    fetch_team_assets(team_names)


def _job_openfootball_refresh() -> None:
    """Pull the canonical group letters from openfootball/world-cup."""
    fetch_cup_txt()


def _job_fifa_ranking_refresh() -> None:
    fetch_fifa_ranking()


def _job_wikipedia_squads_refresh() -> None:
    """One-shot squad pull — invoked manually via /_ops/run-job.

    The team→page-title map lives in ``data/wc2026_squad_pages.json``; we keep
    it out of code so the operator can edit it without a deploy when Wikipedia
    renames a page mid-tournament.
    """
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    cfg = Path("data/wc2026_squad_pages.json")
    if not cfg.exists():
        logger.warning("data/wc2026_squad_pages.json missing — skipping squad ingest")
        return
    team_pages = json.loads(cfg.read_text(encoding="utf-8"))
    fetch_all_squads(team_pages)


def _job_football_data_co_uk_refresh() -> None:
    """Pull the default club-league closing-odds calibration corpus."""
    fetch_calibration_corpus()


def _job_fbref_refresh() -> None:
    """Pull FBref match-log xG for the 48 WC teams.

    The team→URL map lives in ``data/wc2026_fbref_pages.json`` (same operator-
    editable pattern as the Wikipedia squad pages).
    """
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    cfg = Path("data/wc2026_fbref_pages.json")
    if not cfg.exists():
        logger.warning("data/wc2026_fbref_pages.json missing — skipping FBref ingest")
        return
    raw = json.loads(cfg.read_text(encoding="utf-8"))
    team_pages = [(team, url) for team, url in raw.items()]
    fetch_team_match_logs(team_pages)


def _job_statsbomb_refresh() -> None:
    """Pull StatsBomb open-data shots for the 4 men's tournaments + refit xG model.

    StatsBomb open data is effectively immutable once a tournament is in the
    archive, so this job is manual-only — re-fetching daily would waste
    bandwidth. After the corpus is on disk it also refits the xG shot model.
    """
    paths = fetch_all_tournament_shots()
    if not paths:
        logger.warning("statsbomb refresh: no shots fetched, skipping xG refit")
        return
    from wc2026.ingest.statsbomb_open import load_shots_corpus  # noqa: PLC0415
    from wc2026.models.xg_shot_model import fit_and_save  # noqa: PLC0415

    corpus = load_shots_corpus()
    if corpus.empty:
        logger.warning("statsbomb refresh: empty corpus on disk, skipping xG refit")
        return
    fit_and_save(corpus)


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


def _job_xgb_refit() -> None:
    """Refit the Phase 5 XGB H/D/A classifier on the latest training corpus.

    Weekly cadence: the underlying corpus only updates after fresh ingest
    runs, and the XGB is a slow learner — daily refit would burn cycles with
    little change. After the artefact is written, restart the API to pick it
    up (the lifespan loader caches XGB + SHAP explainer at startup).
    """
    from scripts.refit_xgb import refit_and_save  # noqa: PLC0415

    refit_and_save()


def _job_features_rebuild() -> None:
    """Rebuild the materialised features_match_features table.

    Also writes a daily WC 2026 prediction snapshot to ``model_predictions``
    against the same freshly-fit PoissonDC artefact — Phase 7's rolling
    calibration job (``/api/v1/track-record/wc2026``) reads from that table.
    Both steps share the same DB dependency, so we run them together.

    Runs after ``poisson_refit`` so the Poisson outputs in each row match the
    freshly-fit artefact. Skipped (with a logged warning) when no
    ``DATABASE_URL`` / ``WC2026_DATABASE_URL`` is set — without a DB there's
    nowhere to persist to.
    """
    if not (os.environ.get("DATABASE_URL") or os.environ.get("WC2026_DATABASE_URL")):
        logger.warning(
            "neither DATABASE_URL nor WC2026_DATABASE_URL set — skipping features rebuild"
        )
        return
    from scripts.build_features import build_and_persist_features  # noqa: PLC0415
    from scripts.persist_wc2026_predictions import persist_daily_snapshot  # noqa: PLC0415

    build_and_persist_features()
    persist_daily_snapshot()


def _job_monte_carlo_rerun() -> None:
    """Re-run the 10k Monte Carlo conditioned on completed-match results.

    Registered only during the WC 2026 window. Skipped when no DB is
    configured (the persistence script handles the env check itself).
    """
    today = datetime.now(UTC).date()
    if not (WC_TOURNAMENT_START <= today <= WC_TOURNAMENT_END):
        return
    from scripts.rerun_monte_carlo import rerun_and_persist  # noqa: PLC0415

    rerun_and_persist()


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
    JobSpec(name="features_rebuild", hour=5, minute=15, func=_job_features_rebuild),
    # Weekly Sunday 03:00 — TheSportsDB metadata + openfootball cup.txt.
    JobSpec(
        name="thesportsdb_refresh",
        hour=3,
        minute=0,
        func=_job_thesportsdb_refresh,
        day_of_week="sun",
    ),
    JobSpec(
        name="openfootball_refresh",
        hour=3,
        minute=30,
        func=_job_openfootball_refresh,
        day_of_week="sun",
    ),
    # Monthly 1st 06:00 — FIFA Men's World Ranking snapshot.
    JobSpec(
        name="fifa_ranking_refresh",
        hour=6,
        minute=0,
        func=_job_fifa_ranking_refresh,
        day=1,
    ),
    # Weekly Sunday 03:45 — football-data.co.uk closing-odds calibration corpus.
    JobSpec(
        name="football_data_co_uk_refresh",
        hour=3,
        minute=45,
        func=_job_football_data_co_uk_refresh,
        day_of_week="sun",
    ),
    # Weekly Sunday 05:30 — FBref match-log xG for the 48 WC teams.
    JobSpec(
        name="fbref_refresh",
        hour=5,
        minute=30,
        func=_job_fbref_refresh,
        day_of_week="sun",
    ),
    # Weekly Sunday 05:45 — XGB H/D/A classifier refit (runs after the daily
    # poisson_refit at 05:00 + features_rebuild at 05:15 + fbref at 05:30).
    JobSpec(
        name="xgb_refit",
        hour=5,
        minute=45,
        func=_job_xgb_refit,
        day_of_week="sun",
    ),
)


MANUAL_ONLY_JOB_SPECS: tuple[JobSpec, ...] = (
    # Squads update on no regular cadence; coaches finalise rosters at irregular
    # intervals before the tournament. Triggered manually from the Operator page.
    JobSpec(
        name="wikipedia_squads_refresh",
        hour=-1,
        minute=-1,
        func=_job_wikipedia_squads_refresh,
    ),
    # StatsBomb open data is effectively immutable per-tournament; manual-only.
    JobSpec(
        name="statsbomb_refresh",
        hour=-1,
        minute=-1,
        func=_job_statsbomb_refresh,
    ),
)


JOB_REGISTRY: dict[str, JobSpec] = {
    spec.name: spec for spec in (*JOB_SPECS, *MANUAL_ONLY_JOB_SPECS)
}


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
        trigger = CronTrigger(
            hour=spec.hour,
            minute=spec.minute,
            day_of_week=spec.day_of_week,
            day=spec.day,
            timezone="UTC",
        )
        scheduler.add_job(
            _wrap_with_tracking(spec),
            trigger=trigger,
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
        # Phase 8: conditional Monte Carlo rerun every 30 min during the
        # tournament. When the live poller writes a FT_WHISTLE row, the next
        # firing folds it into the simulator's known-results map and persists
        # the freshly-conditioned standings to tournament_sim_runs.
        mc_spec = JobSpec(
            name="monte_carlo_rerun",
            hour=-1,
            minute=-1,
            func=_job_monte_carlo_rerun,
        )
        scheduler.add_job(
            _wrap_with_tracking(mc_spec),
            trigger=IntervalTrigger(
                minutes=MONTE_CARLO_RERUN_INTERVAL_MINUTES, timezone="UTC"
            ),
            id=mc_spec.name,
            name=mc_spec.name,
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
    "JOB_REGISTRY",
    "JOB_SPECS",
    "MANUAL_ONLY_JOB_SPECS",
    "STANDINGS_WARM_INTERVAL_MINUTES",
    "WC_TOURNAMENT_END",
    "WC_TOURNAMENT_START",
    "JobSpec",
    "build_scheduler",
    "is_tournament_window",
    "main",
    "register_jobs",
]
