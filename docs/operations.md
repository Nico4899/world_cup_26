# Per-matchday operations runbook

Maps each tournament-window check to an Operator-page action or a CLI
command. Print this and tape it to the wall for June 11 → July 19 2026.

The Operator dashboard at `/ops` surfaces every status row that follows.
The CLI commands assume `uv` for Python and `fly` for prod.

## Pre-match (T - 4 h)

| Check | Where | OK if… | If not OK |
|---|---|---|---|
| Model fit at | `/ops` "Model fit at" tile | < 24 h old | `fly ssh console -C "python -m scripts.refit_poisson_dc"` then `fly machines restart` |
| Elo snapshot age | `/ops` "Elo snapshot age" tile | ≤ 2 days | Trigger `elo_refresh` from `/ops`; if it errors twice, see "scraper broken" below |
| Group letters | `/ops` "Group letters" tile | "FIFA draw" once the official JSON is dropped at `data/wc2026_group_assignment.json`; "derived" is acceptable before then | Drop the JSON in place; restart the API |
| Scheduler heartbeat | `/ops` scheduler table | Every daily job has a row in the past 24 h with status = `ok` | Re-run via the manual trigger; if it errors, check `/health` for the underlying cause |
| Live event poller wired | `/ops` scheduler table | `live_events_poll` row exists with status `ok` (tournament window only) | If `FOOTBALL_DATA_ORG_KEY` is unset, `fly secrets set` it; otherwise restart the scheduler |

## During the match (T 0)

| Check | Where | OK if… |
|---|---|---|
| Live SSE connects | `/match/{id}` shows "🔴 LIVE" badge + per-event line chart updates | Win-prob line extends every ~5 s |
| Score matches reality | `/match/{id}` snapshot line vs. an external source (TheSportsDB, FIFA.com) | Same scoreline |
| `/live/{id}/sse` stream healthy | `/_ops/scheduler-status` shows recent `live_events_poll` rows | Last row < 60 s old |

If `/match/{id}` shows "🔴 LIVE" with stale data: the SSE stream is fine
but the upstream poller is stuck. Re-trigger `live_events_poll` from `/ops`
or `fly ssh console -C "python -c 'from wc2026.scheduler.jobs import
_job_live_events_poll; _job_live_events_poll()'"`.

## Post-match (T + 30 min)

| Check | Where | OK if… |
|---|---|---|
| FT_WHISTLE row written | `/match/{id}` shows "✅ FULL TIME" | Status changes within 5 min of the actual whistle |
| Track Record picks the match up | `/track-record` "Completed matches" counter increments | Counter matches the number of finished WC 2026 fixtures |
| Monte Carlo rerun fired | `/groups` provenance caption shows "persisted run #N" where N changed | New run id within 30 min of the FT_WHISTLE |
| Match prediction snapshot persisted | DB `model_predictions` row count grew | One additional row per WC 2026 fixture per refit |

## End of matchday (T + 2 h)

| Check | Where | OK if… |
|---|---|---|
| Calibration not degenerate | `/track-record` log-loss vs. baseline_log_loss | model log-loss < baseline (negative delta) |
| Reliability scatter on-trend | `/track-record` historical scatter | Points clustered along the y = x reference |
| pg_dump uploaded | Cloudflare R2 console / S3 console | New `.sql.gz` for today's date |

## Daily (cron 02:00 UTC)

`db_backup` runs automatically:

```bash
# Verify the latest dump landed in R2
fly ssh console -C "ls -la /app/data/backups | tail"

# Or against R2 directly:
aws s3 ls s3://wc2026-backups/wc2026/backups/ \
  --endpoint-url https://<account-id>.r2.cloudflarestorage.com | tail
```

## Failure modes

### eloratings.net scraper broken

`elo_refresh` failing repeatedly with a parse error usually means the site
markup changed. While the fix is in flight:

1. Check `/ops` → `elo_refresh` last error.
2. Apply a manual override via `POST /api/v1/_ops/elo-override` (see the
   endpoint docs at `/docs#/ops/post_ops_elo_override`).
3. Once the scraper is fixed, the next successful `elo_refresh` overwrites
   any stale rows; the override table stays as the source of truth until
   manually cleared.

### football-data.org rate-limited (HTTP 429)

`live_events_poll` returning 429s during a live match.

1. Confirm the limit: `/ops` → `live_events_poll` last error.
2. Halve the poll cadence: `fly secrets set WC2026_LIVE_POLL_INTERVAL=120`
   then `fly machines restart`.
3. If the same again, escalate by switching to TheSportsDB as the secondary
   feed for the live snapshot (the poller already takes both sources).

### SSE drops

Browser tab shows "Live stream disconnected — retrying."

- Single tab: the client-side `useLiveWinProb` hook reconnects with
  exponential backoff up to 30 s. No operator action needed.
- All tabs: the API is unreachable. Run `curl https://<your-app>.fly.dev/health`
  to confirm and `fly status` to see which machine is unhealthy.

### Streamlit Cloud is dead (sleep)

There is no Streamlit deploy; the dashboard is on Vercel and doesn't sleep.

### Vercel deploy errors

Vercel emails the failure. The most likely cause is the methodology MDX
sync — check the prebuild step. Re-run by triggering an empty commit or
clicking "Redeploy" on Vercel.

## Manual triggers worth knowing

All from `/ops` → "Manual triggers":

| Job | When to fire |
|---|---|
| `poisson_refit` | If `/health` shows model_fitted = false |
| `monte_carlo_rerun` | If `/groups` provenance shows a stale run id post-match |
| `live_events_poll` | If `/match/{id}` is mid-match but the score is stale > 60 s |
| `wikipedia_squads_refresh` | Once FIFA publishes final squads (mid-May 2026 per past tournaments) |
| `db_backup` | Before destructive operations (schema migrations, ops experiments) |

## CLI cheats (when the API is unreachable)

```bash
# Tail scheduler logs
fly logs -i scheduler

# Re-fit PoissonDC from scratch
fly ssh console -C "python -m scripts.refit_poisson_dc"

# Trigger conditional MC out of band
fly ssh console -C "python -m scripts.rerun_monte_carlo --n-sims=10000"

# Restore from the latest pg_dump
fly ssh console -C "gunzip -c /app/data/backups/$(ls -t /app/data/backups | head -1) | psql \$DATABASE_URL"
```
