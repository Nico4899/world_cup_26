# Architecture

A Python backend (FastAPI + APScheduler + Postgres on Fly.io) and a
**Next.js 16 frontend on Vercel** share the same repo. Three Python runtime
processes (API, scheduler, Postgres) talk to one PostgreSQL database; the
Next.js app runs separately and consumes the FastAPI over HTTPS. All data
lives on disk or in Postgres; nothing in-process is load-bearing for
restart safety.

Built over 11 Stage 2 phases on the Stage 1 baseline, plus a Next.js
migration (Phases A-H) that replaced the Streamlit dashboard with a
Next.js 16 + React 19 + Visx UI deployed to Vercel. See
[`README.md#stage-2-roadmap-complete`](../README.md#stage-2-roadmap-complete)
for the phase-by-phase log; this doc shows the **current as-built**.

## High-level (the whole platform on one diagram)

```mermaid
flowchart LR
    subgraph SRC[Free data sources — 9]
        K[Kaggle: Jürisoo intl]
        E[eloratings.net]
        F[football-data.org]
        TS[TheSportsDB]
        OF[openfootball]
        WP[Wikipedia / Wikidata]
        SB[StatsBomb open]
        FB[FBref]
        FDC[football-data.co.uk]
    end

    subgraph SCH[Scheduler — APScheduler]
        DAILY[5 daily cron + monthly + 3 weekly]
        WIN[3 interval jobs<br/>tournament-window only]
        MAN[2 manual-only]
    end

    subgraph DB[PostgreSQL — 11 tables]
        RAW[(raw_*: matches, elo_snapshots,<br/>team_assets, squads, fifa_rankings,<br/>match_xg, live_events)]
        FEAT[(features_match_features)]
        MOD[(model_predictions)]
        SIM[(tournament_sim_runs<br/>+ team_outcomes)]
        JOBS[(scheduler_job_runs)]
    end

    subgraph DISK[On-disk artefacts]
        APD[poisson_dc/latest.npz]
        ASH[shootout/latest.json]
        AXG[xg_shot/latest.json]
        AXM[xgb/latest.json + meta]
        ALW[live_win_prob/latest.json]
    end

    subgraph ML[Model + simulator]
        P[PoissonDC + Dixon-Coles]
        SHO[Shootout submodel]
        XGS[xG shot model]
        XGB[XGBoost H/D/A + SHAP]
        BL[Geometric blend]
        LWP[Live win-prob]
        MC[Monte Carlo<br/>conditional after FT_WHISTLE]
    end

    subgraph API[FastAPI — 25 endpoints]
        BASE[Stage 1: matches / predictions /<br/>tournament / teams / h2h / health]
        P5[Phase 5: /explain]
        P6[Phase 6: /live + /sse]
        P7[Phase 7: /track-record]
        P9[Phase 9: /teams/.../elo-history,<br/>tournament-probs, assets,<br/>fifa-rankings, squad, xg-form]
        OPS[/_ops: scheduler-status,<br/>available-jobs, run-job]
    end

    subgraph UI[Next.js on Vercel — 9 routes]
        TODAY[/ Today]
        MD[/match/[id]<br/>SHAP popovers + live SSE chart]
        GRP[/groups<br/>5-segment Visx bars]
        BR[/bracket<br/>scenarios + conditional locks]
        TR[/track-record<br/>+ historical reliability scatter]
        AB[/about<br/>MDX methodology]
        OP[/ops<br/>Server Actions for run-job]
        TP[/team/[name]<br/>path-to-final + xG splits]
        MAP[/map<br/>deck.gl + MapLibre]
    end

    subgraph OPS_LAYER[Phase 10 observability]
        SE[Sentry SDK]
        S3R[S3 / R2 backup]
    end

    SRC --> SCH --> RAW
    SCH -.refit.-> P --> APD
    SCH -.refit.-> SHO --> ASH
    SCH -.refit.-> XGS --> AXG
    SCH -.refit.-> XGB --> AXM
    SCH -.refit.-> LWP --> ALW
    SCH -.rebuild.-> FEAT
    SCH -.snapshot.-> MOD
    SCH -.conditional rerun.-> SIM
    SCH -.daily backup.-> DB --> S3R
    SCH -.logs.-> JOBS

    DISK --> ML
    DB --> ML
    ML --> MC
    MC --> SIM

    DB --> API
    DISK --> API
    ML --> API
    API --> UI
    SE -.errors.-> API
    SE -.errors.-> SCH
```

## Process responsibilities

| Service | Module | Purpose | Restart safety |
|---|---|---|---|
| FastAPI | `src/wc2026/api/main.py` | 26 endpoints across 11 routers; loads PoissonDC + shootout + XGB + SHAP + live-win-prob + Elo snapshot in lifespan | Stateless — every artefact reloaded on restart |
| Next.js (Vercel) | `frontend/src/app/` | 9 App Router routes; data via TanStack Query (client) + Next `fetch` with `revalidate` (server). Server Actions inject `WC2026_OPS_TOKEN` from Vercel env for manual job triggers | Stateless; Vercel rebuilds on every push |
| Scheduler | `src/wc2026/scheduler/jobs.py` | 13 cron jobs + 3 interval-triggered tournament-window jobs + 2 manual-only; logs each run to `scheduler_job_runs` for the Operator page | Re-registers triggers on startup; missed runs skipped (no catch-up) |

## Scheduled jobs (full list)

### Daily / weekly / monthly cron

| Job | Cadence | What it does |
|---|---|---|
| `db_backup` | daily 02:00 UTC | `pg_dump` → local volume → S3/R2 upload + 30-day remote prune |
| `thesportsdb_refresh` | Sunday 03:00 UTC | crest / kit / stadium metadata for the 48 WC teams |
| `openfootball_refresh` | Sunday 03:30 UTC | canonical group letters from openfootball/world-cup |
| `football_data_co_uk_refresh` | Sunday 03:45 UTC | closing-odds calibration corpus |
| `kaggle_refresh` | daily 04:00 UTC | Jürisoo intl results CSV → Parquet snapshot |
| `elo_refresh` | daily 04:15 UTC | eloratings.net two-TSV polite scrape |
| `football_data_org_refresh` | daily 04:30 UTC | WC 2026 fixtures + scores |
| `poisson_refit` | daily 05:00 UTC | refit PoissonDC + shootout submodel |
| `features_rebuild` | daily 05:15 UTC | rebuild `features_match_features` + persist daily WC 2026 prediction snapshot to `model_predictions` |
| `fbref_refresh` | Sunday 05:30 UTC | FBref match-log xG for the 48 teams |
| `xgb_refit` | Sunday 05:45 UTC | refit Phase 5 XGBoost classifier |
| `fifa_ranking_refresh` | 1st of month 06:00 UTC | FIFA Men's World Ranking snapshot |

### Interval-triggered (tournament window only: 2026-06-11 → 2026-07-19)

| Job | Cadence | What it does |
|---|---|---|
| `standings_cache_warm` | every 60 min | force the API to recompute the in-process standings cache |
| `monte_carlo_rerun` | every 30 min | re-run 10 k Monte Carlo conditioned on completed-match results → `tournament_sim_runs` |
| `live_events_poll` | every 60 s | pull today's WC fixtures, call `poll_live_match` for each IN_PLAY / PAUSED / recently FINISHED |

### Manual-only

| Job | Trigger | What it does |
|---|---|---|
| `wikipedia_squads_refresh` | `POST /api/v1/_ops/run-job/wikipedia_squads_refresh` | tournament squad rosters |
| `statsbomb_refresh` | same | StatsBomb shots corpus + xG shot model refit + live win-prob model refit |

## API surface (26 endpoints, 11 routers)

| Router | Endpoints |
|---|---|
| `health` | `GET /health` |
| `matches` | `GET /api/v1/matches`, `GET /api/v1/matches/{id}` |
| `predictions` | `GET /api/v1/predictions/{home}/{away}` (with `?blend=true`) |
| `tournament` | `GET /standings`, `GET /bracket`, `GET /groups-live`, `POST /bracket/conditional`, `/teams/{team}/path-to-final` |
| `teams` | `GET /{team}/recent`, `/{team}/elo-history`, `/{team}/tournament-probs`, `/{team}/assets`, `/{team}/fifa-rankings`, `/{team}/squad`, `/{team}/xg-form`, `/{team}/path-to-final` |
| `h2h` | `GET /api/v1/h2h/{a}/{b}` |
| `explain` | `GET /api/v1/explain/{match_id}` (Phase 5 SHAP; 503 when no XGB artefact) |
| `live` | `GET /api/v1/live/{id}`, `/history`, `/sse` (Phase 6) |
| `track_record` | `GET /api/v1/track-record/wc2026`, `GET /api/v1/track-record/historical/{tournament}` (24h cache) |
| `ops` | `GET /_ops/scheduler-status`, `/_ops/available-jobs`, `POST /_ops/run-job/{name}` |

## Data flow guarantees

- **Ingest is idempotent.** Re-running any ingester overwrites Parquet snapshots; Postgres upserts use `ON CONFLICT DO NOTHING`/`DO UPDATE`. No duplicate rows.
- **Model fit is deterministic.** Same input matches + same weights + same `ref_date` → identical PoissonDC parameters (scipy `L-BFGS-B`, no random seed).
- **Monte Carlo is seeded.** Same seed + same model + same `known_group_results` → identical tournament. Cached in the API by `(n_sims, seed)`; the Phase 8 conditional rerun persists 1 run per ~30 min during the tournament.
- **Live polling is rate-bounded.** `live_events_poll` runs every 60 s during the tournament; the football-data.org cost is ≤4 calls/min on the busiest matchday vs. the 10/min free-tier ceiling.
- **Sentry + S3 backup are env-gated.** No `SENTRY_DSN` / `AWS_S3_BUCKET` → silent no-op; the system still works locally with neither.

## What is intentionally not here (honest scope)

- **Transfermarkt squad market value** — fragile, legal grey area. Squad age from Wikipedia suffices.
- **FIFA.com unofficial JSON** — T&Cs prohibit redistribution; openfootball + football-data.org cover the fixture/group needs.
- **Paid data feeds** (Opta, Sportradar, Enetpulse, paid StatsBomb) — free corpus is sufficient for tournament-level prediction.
- **Bidirectional WebSockets** — SSE is one-way (server → client). The dashboard polls + SSE-consumes, never sends.
- **Prometheus + Grafana stack** — Sentry + `/health` + `/api/v1/_ops/scheduler-status` cover monitoring at this scale.
- **Click-to-set SVG bracket** — Tier 3 shipped a Python-friendly equivalent: per-match lock form on `/bracket` driving `POST /api/v1/tournament/bracket/conditional`. A drag-and-drop SVG bracket would be a polish lever, not a feature gap.
- **Live red-card / sub tracking** — football-data.org's free tier exposes only the score, not detailed events. Phase 6 acknowledges the limitation in `ingest/live_events.py`.

## Where to look for X

| If you want… | Look here |
|---|---|
| The end-to-end deploy runbook | [`docs/deploy.md`](deploy.md) |
| The model methodology + references | [`docs/methodology.md`](methodology.md) |
| Per-source licence terms | [`docs/LICENSES.md`](LICENSES.md) |
| Operational status (job runs, freshness, manual triggers) | the **Operator** dashboard page |
| Live tournament calibration (Brier / log-loss / RPS as matches finish) | the **Track Record** dashboard page |
| The plan that drove Stage 2 | `/Users/nico/.claude/plans/extensively-review-the-given-resilient-anchor.md` |
