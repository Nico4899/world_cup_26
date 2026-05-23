"""Operator-facing endpoints: scheduler-job status + manual triggers.

Mounted under ``/api/v1/_ops`` so the underscore prefix flags "not a
public/customer-facing path" in route listings.

The manual-trigger endpoint (``POST /run-job/{name}``) requires the
``X-Ops-Token`` header to match the ``WC2026_OPS_TOKEN`` env var when that
env var is set; if it is unset the endpoint is open (acceptable for local
development; production deployments must set the token).

If the DB is unreachable the read endpoint returns a 503 with an explanatory
body — operators expect "did the scheduler run?" to be informative even when
Postgres is the thing that's broken.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Path
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.exc import SQLAlchemyError

from wc2026.db.models import SchedulerJobRun
from wc2026.db.session import session_scope
from wc2026.scheduler.jobs import JOB_REGISTRY, _record_job_run

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/_ops")

ENV_OPS_TOKEN = "WC2026_OPS_TOKEN"


class JobRunRow(BaseModel):
    job_name: str
    last_run_at: datetime
    last_status: str = Field(description="ok | error")
    last_error_text: str | None = None


class SchedulerStatusResponse(BaseModel):
    jobs: list[JobRunRow] = Field(
        description="One row per distinct job_name (the most recent run for that job)."
    )


class RunJobResponse(BaseModel):
    job_name: str
    enqueued_at: datetime
    status: str = Field(description="'enqueued' — see /scheduler-status for the outcome.")


def _check_ops_token(provided: str | None) -> None:
    """Compare X-Ops-Token to the env var when set; no-op when unset."""
    expected = os.environ.get(ENV_OPS_TOKEN)
    if not expected:
        return
    if not provided or provided != expected:
        raise HTTPException(status_code=403, detail="bad or missing X-Ops-Token")


def _run_job_safely(job_name: str) -> None:
    """Run a registered job and record its outcome in scheduler_job_runs.

    Intended to be enqueued via FastAPI ``BackgroundTasks``; never raises so
    a slow/failed job doesn't crash the worker.
    """
    spec = JOB_REGISTRY.get(job_name)
    if spec is None:
        logger.warning("manual-trigger asked for unknown job %r", job_name)
        return
    started = datetime.now(UTC)
    try:
        spec.func()
    except Exception as exc:
        logger.exception("Manual run of job %s failed", job_name)
        _record_job_run(job_name, started, "error", repr(exc))
        return
    _record_job_run(job_name, started, "ok", None)


@router.get("/scheduler-status", response_model=SchedulerStatusResponse)
def scheduler_status() -> SchedulerStatusResponse:
    """Return the latest ``scheduler_job_runs`` row per ``job_name``."""
    try:
        with session_scope() as s:
            # Pull the most recent run per job_name. Cheap: tens of rows max.
            rows = s.execute(
                select(SchedulerJobRun).order_by(desc(SchedulerJobRun.started_at))
            ).scalars()
            seen: dict[str, JobRunRow] = {}
            for r in rows:
                if r.job_name in seen:
                    continue
                seen[r.job_name] = JobRunRow(
                    job_name=r.job_name,
                    last_run_at=r.started_at,
                    last_status=r.status,
                    last_error_text=r.error_text,
                )
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503, detail=f"Scheduler-status DB query failed: {exc.__class__.__name__}"
        ) from exc
    return SchedulerStatusResponse(jobs=sorted(seen.values(), key=lambda r: r.job_name))


class AvailableJobsResponse(BaseModel):
    jobs: list[str] = Field(description="Names of every job that can be invoked via /run-job.")


@router.get("/available-jobs", response_model=AvailableJobsResponse)
def available_jobs() -> AvailableJobsResponse:
    """List every job name the operator UI may surface as a manual trigger."""
    return AvailableJobsResponse(jobs=sorted(JOB_REGISTRY.keys()))


@router.post("/run-job/{job_name}", response_model=RunJobResponse, status_code=202)
def run_job(
    background_tasks: BackgroundTasks,
    job_name: str = Path(..., description="A name from /api/v1/_ops/available-jobs"),
    x_ops_token: str | None = Header(default=None),
) -> RunJobResponse:
    """Enqueue a registered job to run in the background.

    Returns 202 immediately; the actual outcome (ok / error) shows up in
    ``/api/v1/_ops/scheduler-status`` after the job completes.
    """
    _check_ops_token(x_ops_token)
    if job_name not in JOB_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"unknown job {job_name!r}; see /api/v1/_ops/available-jobs",
        )
    background_tasks.add_task(_run_job_safely, job_name)
    return RunJobResponse(
        job_name=job_name,
        enqueued_at=datetime.now(UTC),
        status="enqueued",
    )
