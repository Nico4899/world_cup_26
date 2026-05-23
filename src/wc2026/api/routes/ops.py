"""Operator-facing endpoints: scheduler-job status, etc.

Read-only; mounted under ``/api/v1/_ops`` so the underscore prefix flags
"not a public/customer-facing path" in route listings.

If the DB is unreachable the endpoint returns a 503 with an explanatory body —
operators expect "did the scheduler run?" to be informative even when Postgres
is the thing that's broken.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.exc import SQLAlchemyError

from wc2026.db.models import SchedulerJobRun
from wc2026.db.session import session_scope

router = APIRouter(prefix="/api/v1/_ops")


class JobRunRow(BaseModel):
    job_name: str
    last_run_at: datetime
    last_status: str = Field(description="ok | error")
    last_error_text: str | None = None


class SchedulerStatusResponse(BaseModel):
    jobs: list[JobRunRow] = Field(
        description="One row per distinct job_name (the most recent run for that job)."
    )


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
