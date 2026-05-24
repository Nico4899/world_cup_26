# Deploy runbook

End-to-end checklist for deploying `wc2026-predictor` to Fly.io with off-site
backups (Cloudflare R2) and error monitoring (Sentry). The Next.js frontend
lands on Vercel and points at the Fly API.

## Pre-requisites

- `flyctl` installed and authenticated (`fly auth login`)
- A Cloudflare account (R2 is recommended; 10 GB free tier, no egress fees)
  *or* an AWS account if you prefer S3
- A Sentry account with a free-tier project (~5 GB events/month)
- A Vercel account (Hobby plan suffices; auths via GitHub OAuth)

## Phase 10A — Fly API + scheduler + Postgres

### 1. App + Postgres + volume

```bash
cp fly.toml.example fly.toml
# Edit `app =` and `primary_region =` to your preferred values.

fly launch --no-deploy --copy-config

fly volumes create wc2026_data --size 2 --region <your-region>
fly postgres create --name wc2026-pg --region <your-region>
fly postgres attach wc2026-pg     # injects DATABASE_URL into the app secrets
```

Cost: smallest Postgres + two `shared-cpu-1x` Machines + 2 GB volume ≈ **$5–10/month** on the hobby tier.

### 2. Secrets

Every secret below is **optional** — missing values gracefully degrade the
corresponding feature, never blocking startup.

```bash
fly secrets set \
  FOOTBALL_DATA_ORG_KEY=<your-fdo-key> \
  SENTRY_DSN=<your-sentry-dsn> \
  AWS_S3_BUCKET=<your-bucket> \
  AWS_S3_ENDPOINT_URL=<your-endpoint> \
  AWS_ACCESS_KEY_ID=<your-access-key-id> \
  AWS_SECRET_ACCESS_KEY=<your-secret>
```

Verify with `fly secrets list` (values are not shown, only names + digests).

### 3. Initial deploy

```bash
fly deploy
fly scale count app=1 scheduler=1
fly status
```

The app process exposes `/health` on the Fly internal port; the scheduler
process runs APScheduler with the daily cron + tournament-window interval
jobs. Both pull the same Docker image (`Dockerfile.app`).

### 4. First-time DB migration

```bash
fly ssh console -C "alembic upgrade head"
```

Idempotent — re-running is safe.

### 5. Smoke check

```bash
curl https://<your-app>.fly.dev/health
# expect: {"status": "ok", "model_fitted": true, ...}
```

If `model_fitted` is `false`: trigger a manual refit via
`fly ssh console -C "python -m scripts.refit_poisson_dc"`, then `fly machines restart`.

## Phase 10B — Cloudflare R2 (off-site backup)

R2 is S3-compatible; the same `boto3` client works against either by setting
`AWS_S3_ENDPOINT_URL` appropriately.

### 1. Bucket + API token

In the Cloudflare dashboard:
1. Create an R2 bucket named (e.g.) `wc2026-backups`.
2. Create an R2 API token with **Object Read & Write** scoped to that bucket.
3. Note the `Account ID` and the generated `Access Key ID` + `Secret Access Key`.

### 2. Endpoint URL

R2's S3 endpoint is `https://<account-id>.r2.cloudflarestorage.com`.

### 3. Set Fly secrets

```bash
fly secrets set \
  AWS_S3_BUCKET=wc2026-backups \
  AWS_S3_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com \
  AWS_ACCESS_KEY_ID=<r2-access-key> \
  AWS_SECRET_ACCESS_KEY=<r2-secret>
```

The daily `db_backup` job (02:00 UTC) writes a gzipped `pg_dump` to the local
volume, then uploads it to R2 and prunes anything older than 30 days.

### 4. Verify

After the first scheduled run (next 02:00 UTC), check R2:

```bash
aws s3 ls s3://wc2026-backups/wc2026/backups/ \
  --endpoint-url https://<account-id>.r2.cloudflarestorage.com
```

A new `.sql.gz` should appear daily. Manual upload also works:

```bash
fly ssh console -C "python -m scripts.refit_poisson_dc"
fly ssh console -C "python -c 'from wc2026.scheduler.jobs import _job_db_backup; _job_db_backup()'"
```

## Phase 10C — Sentry

### 1. Create the project

In Sentry: New Project → Platform: FastAPI → Project name: `wc2026-api`. Note the DSN.

### 2. Set the Fly secret

```bash
fly secrets set SENTRY_DSN=<dsn>
```

Both the API process and the scheduler call `init_sentry(service=…)` at boot,
so errors from either show up tagged accordingly.

### 3. Verify

Trigger a synthetic error to confirm wiring:

```bash
fly ssh console -C "python -c 'import sentry_sdk; sentry_sdk.init(\"<dsn>\"); 1/0'"
```

The exception should appear in Sentry within ~30 s.

## Phase 10D — Vercel (Next.js dashboard)

The dashboard is a Next.js 16 / React 19 app under [`frontend/`](../frontend/)
that talks to the Fly API over HTTPS. Vercel handles SSR + edge caching for
the static routes (Bracket, Map) and request-time fetches for the dynamic
ones (Today, Groups, Match Detail, Team Profile, Track Record, Ops).

### 1. Push to GitHub

Vercel deploys from a GitHub-hosted repo. Make sure your branch is pushed
before the next step.

### 2. Create the Vercel project

In [vercel.com/new](https://vercel.com/new) → Import Git Repository:
- Repository: `Nico4899/world_cup_26`
- Framework Preset: **Next.js** (auto-detected from `frontend/package.json`)
- Root Directory: `frontend`
- Build Command: leave default (`pnpm build` — the `prebuild` script runs
  `node scripts/sync-methodology.mjs` automatically)
- Install Command: leave default (Vercel detects pnpm via `pnpm-lock.yaml`)

### 3. Environment variables

In Project Settings → Environment Variables:

| Name | Value | Scope |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `https://<your-fly-app>.fly.dev` | Production + Preview |
| `WC2026_OPS_TOKEN` | _same value set on Fly_ | Production only |

The `WC2026_OPS_TOKEN` is **server-only** (no `NEXT_PUBLIC_` prefix). The
Operator page's "Run job" buttons are Server Actions that attach the token
as `X-Ops-Token` server-side; the browser never sees it.

### 4. Allow Vercel origins in the API CORS list

The Fly API already allows `https://*.vercel.app` for preview deployments
via [`src/wc2026/api/main.py`](../src/wc2026/api/main.py)'s
`CORS_ALLOW_ORIGIN_REGEX`. For the production custom domain (e.g.
`wc2026.example.com`), set:

```bash
fly secrets set WC2026_FRONTEND_ORIGIN=https://wc2026.example.com
fly machines restart
```

### 5. Verify

After the first deploy, walk through:
- `https://<your-vercel-app>.vercel.app/` — Today's predictions load.
- `/match/0` — Heatmap renders, SHAP popovers open per outcome tile.
- `/ops` — Manual job triggers fire (Server Action attaches `X-Ops-Token`).
- `/api/v1/_ops/scheduler-status` (on Fly) shows the latest `run-job` call.

### 6. Decommission the old Streamlit Community Cloud deploy

Once Vercel goes live and parity is verified, delete the old Streamlit
Cloud app from [share.streamlit.io](https://share.streamlit.io). The
`dashboard/` directory and `Dockerfile.dashboard` were removed in the
same PR that introduced Vercel.

## Cost summary

| Item | Free tier | Paid tier (hobby) |
|---|---|---|
| Fly.io shared-cpu-1x x2 + 2 GB volume + Postgres | — | ~$5–10/mo |
| Cloudflare R2 | 10 GB storage + 1M reads + 10M writes / mo | $0 expected |
| Sentry | 5 GB events / mo | $0 expected |
| Vercel (Hobby) | 100 GB bandwidth, unlimited static, 1M serverless invocations | $0 expected |

Total: **$5–10/month** for the duration of the tournament window. Set
`min_machines_running = 0` in `fly.toml` outside the window and the cost
drops to Postgres + volume only (~$3–5/mo).

## Rollback

```bash
fly releases                          # list deploys with version + timestamp
fly deploy --image-label v<previous>  # roll back to a prior image
```

DB rollbacks: every daily backup is on R2 + the local volume. Restore via:

```bash
fly ssh console -C "gunzip -c /app/data/backups/wc2026-2026-06-20.sql.gz | psql $DATABASE_URL"
```

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `/health` 503 model not loaded | First boot, no artefact on the volume yet | `fly ssh console -C "python -m scripts.refit_poisson_dc"` then `fly machines restart` |
| Scheduler rows missing in `/api/v1/_ops/scheduler-status` | Scheduler process didn't start | `fly status` → if scheduler is unhealthy: `fly machines restart -p scheduler` |
| Live win-prob never updates | `FOOTBALL_DATA_ORG_KEY` unset or rate-limited | `fly secrets list` → re-set if missing; check `/api/v1/_ops/scheduler-status` for `live_events_poll` errors |
| R2 uploads silently skipped | `AWS_S3_BUCKET` unset | `fly secrets set AWS_S3_BUCKET=…` |
| Sentry events not arriving | DSN typo, or sample rate is 0 (default) and you're expecting traces | Set `SENTRY_TRACES_SAMPLE_RATE=0.1` for tracing; errors flow on default settings |
