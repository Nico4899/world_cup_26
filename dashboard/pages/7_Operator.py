"""Operator — at-a-glance health view for the person running the platform.

Shows the data-side facts most useful for "is everything still ticking" —
last-run timestamp per scheduler job, age of the Elo snapshot, when the model
was fit, what flavour of group-letter assignment is in use. Not gated behind a
query param: if you're running this dashboard for yourself, you're also the
operator.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import httpx
import streamlit as st
from dashboard.components.api_client import APIUnreachable, get_json, post_json

st.title("Operator")
st.caption(
    "Operational health and freshness. If these go red, the daily scheduler "
    "(or the ingest sources it depends on) probably stopped — check the "
    "scheduler container logs."
)

# --- /health ----------------------------------------------------------------

try:
    health = get_json("/health")
except APIUnreachable:
    st.error("API is unreachable. Start it with `uv run uvicorn wc2026.api.main:app`.")
    st.stop()

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Model fitted", "yes" if health.get("model_fitted") else "no")
col_a.caption(health.get("model_version") or "version: unknown")

fit_at = health.get("model_fit_at")
fit_at_display = fit_at[:19].replace("T", " ") + " UTC" if fit_at else "—"
col_b.metric("Model fit at", fit_at_display)
col_b.caption(f"{health.get('model_teams_n', 0)} teams in the fitted set")

elo_age = health.get("elo_snapshot_age_days")
elo_date = health.get("elo_snapshot_date") or "—"
if elo_age is None:
    col_c.metric("Elo snapshot", "missing")
    col_c.caption("Shootouts fall back to 50/50.")
else:
    col_c.metric("Elo snapshot age", f"{elo_age} day(s)")
    delta_colour = "off" if elo_age <= 1 else "inverse"
    col_c.caption(
        f"captured {elo_date} · shootouts {'on' if health.get('shootout_model_loaded') else 'OFF'}"
    )
    _ = delta_colour

assignment_source = health.get("group_assignment_source", "derived")
col_d.metric(
    "Group letters", "official" if assignment_source.startswith("official:") else "derived"
)
col_d.caption(
    assignment_source.removeprefix("official:")
    if assignment_source.startswith("official:")
    else "labels derived from fixture dates"
)

# --- /api/v1/_ops/scheduler-status ------------------------------------------

st.subheader("Scheduler jobs")
ops_body = None
try:
    ops_body = get_json("/api/v1/_ops/scheduler-status")
except APIUnreachable:
    st.warning("Scheduler-status endpoint unreachable.")
except httpx.HTTPStatusError as exc:
    # 503 → Postgres is down/unconfigured. Surface it without crashing the page;
    # the operator can still read the model-freshness metrics above.
    if exc.response.status_code == 503:
        st.info(
            "Scheduler-status table is unreachable — Postgres is probably not "
            "running. Bring it up with `docker compose up -d postgres scheduler` "
            "to populate scheduler_job_runs."
        )
    else:
        raise

if ops_body is not None:
    jobs = ops_body.get("jobs", [])
    if not jobs:
        st.info("No scheduler job runs recorded yet — the scheduler may not be running.")
    else:
        now = datetime.now(UTC)
        rows = []
        for j in jobs:
            ran = j["last_run_at"]
            # Parse ISO with offset; fall back to naive parse.
            try:
                ran_dt = datetime.fromisoformat(ran)
            except ValueError:
                ran_dt = None
            age_hours = None
            if ran_dt is not None:
                age_hours = (now - ran_dt).total_seconds() / 3600.0
            rows.append(
                {
                    "Job": j["job_name"],
                    "Last run (UTC)": ran[:19].replace("T", " "),
                    "Age (h)": round(age_hours, 1) if age_hours is not None else "—",
                    "Status": j["last_status"],
                    "Error": (j.get("last_error_text") or "")[:80],
                }
            )
        st.dataframe(rows, hide_index=True, width="stretch")

        stale = [r for r in rows if isinstance(r["Age (h)"], float) and r["Age (h)"] > 36]
        if stale:
            st.warning(
                "Some jobs haven't run in >36h: "
                + ", ".join(r["Job"] for r in stale)
                + ". Daily cron should fire at least every 24h."
            )


# --- Manual triggers --------------------------------------------------------

st.subheader("Manual triggers")
st.caption(
    "Re-run an ingest or refit immediately. The job runs in the background; "
    "outcome shows up in the table above on the next refresh (typically within "
    "30 seconds for cheap jobs, longer for full ingests)."
)

available = None
try:
    available = get_json("/api/v1/_ops/available-jobs")
except APIUnreachable:
    st.info("Manual triggers unavailable while the API is unreachable.")
except httpx.HTTPStatusError:
    st.info("`/available-jobs` endpoint not yet deployed; restart the API.")

if available is not None:
    job_names = sorted(available.get("jobs", []))
    if not job_names:
        st.info("No jobs registered.")
    else:
        # Token entry only shown when the env var is set on the dashboard side;
        # otherwise the API itself decides whether to enforce.
        token_env = os.environ.get("WC2026_OPS_TOKEN")
        token = st.text_input(
            "Ops token (optional; required if the API enforces one)",
            value=token_env or "",
            type="password",
            help="Set WC2026_OPS_TOKEN on the API to require this header.",
        )
        cols = st.columns(min(3, max(1, len(job_names))))
        for i, name in enumerate(job_names):
            with cols[i % len(cols)]:
                if st.button(f"Run {name}", key=f"trigger-{name}"):
                    headers = {"X-Ops-Token": token} if token else None
                    try:
                        body = post_json(
                            f"/api/v1/_ops/run-job/{name}", headers=headers
                        )
                    except APIUnreachable:
                        st.error("API became unreachable mid-request.")
                    except httpx.HTTPStatusError as exc:
                        if exc.response.status_code == 403:
                            st.error("Forbidden — bad or missing X-Ops-Token.")
                        elif exc.response.status_code == 404:
                            st.error(f"Unknown job: {name}")
                        else:
                            st.error(f"HTTP {exc.response.status_code} — {exc.response.text[:200]}")
                    else:
                        st.success(
                            f"Enqueued {body['job_name']} at "
                            f"{body['enqueued_at'][:19].replace('T', ' ')} UTC."
                        )
