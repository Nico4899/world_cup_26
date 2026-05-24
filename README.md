# wc2026-predictor

Calibrated probabilistic predictions for FIFA World Cup 2026.

A FastAPI backend (weighted bivariate Poisson + Dixon–Coles, Monte Carlo
simulator with the official 12-group / 8-best-thirds / R32 bracket) and a
Next.js 16 frontend on Vercel. Honesty about uncertainty is the headline
feature; the backtest gate (WC 2022 log-loss = 1.0379) is non-negotiable.

## Quickstart

Requires Python 3.12 + [`uv`](https://docs.astral.sh/uv/), Node 22 + pnpm,
Docker for local Postgres.

```bash
# 1. Backend
uv sync
docker compose up -d postgres
uv run alembic upgrade head
uv run uvicorn wc2026.api.main:app --port 8000

# 2. Frontend (separate terminal)
cd frontend
pnpm install
NEXT_PUBLIC_API_URL=http://localhost:8000 pnpm dev
```

Then open <http://localhost:3000>.

## Tests + lint

```bash
# Backend
uv run pytest               # 556 unit tests, ~20 s
uv run ruff check .

# Frontend
cd frontend
pnpm typecheck
pnpm lint
pnpm test:run               # Vitest, 26 tests
pnpm e2e                    # Playwright smoke, 10 tests
```

CI runs all of the above on every PR — see
[`.github/workflows/`](.github/workflows/).

## Dashboard routes (9)

| Route | What it shows |
|---|---|
| `/` | Matchday cards with kickoff (UTC), top-3 scorelines, "Why?" popover |
| `/match/[id]` | SHAP popovers per outcome, Visx 6×6 score heatmap, live SSE win-prob chart, Elo-anchored narrative, blend overlay sheet |
| `/groups` | 12 group cards with 5-segment Visx bars, live points + GD, top-10 headline |
| `/bracket` | Single seed / scenarios / **conditional locks** (localStorage-backed, URL-shareable) |
| `/track-record` | Live WC 2026 rolling calibration + WC 2018 / WC 2022 reliability scatter |
| `/about` | Methodology rendered from `docs/methodology.md` via MDX |
| `/ops` | Health, scheduler status, manual job triggers (Server Actions inject `WC2026_OPS_TOKEN` server-side) |
| `/team/[name]` | Elo line, recent form, FIFA ranking, squad, xG splits, MC path-to-final |
| `/map` | 16 host cities on deck.gl + MapLibre, city filter |

## API surface (26 endpoints, 11 routers)

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full table.
Highlights:

- `GET /api/v1/matches/{id}` — fixture + prediction with full 11×11 score matrix
- `GET /api/v1/tournament/standings` — 12-group MC probabilities (persisted run by default)
- `POST /api/v1/tournament/bracket/conditional` — 5k MC with locked knockout winners
- `GET /api/v1/live/{id}/sse` — Server-Sent Events stream of win-prob frames
- `GET /api/v1/explain/{id}` — SHAP top-features (503 when no XGB artefact)
- `GET /api/v1/track-record/historical/{tournament}` — WC 2018 / WC 2022 hindcast (24 h cache)
- `POST /api/v1/_ops/run-job/{name}` — manual scheduler trigger; gated by `X-Ops-Token`

CORS: `localhost:3000` (dev) + `https://*.vercel.app` (previews) +
`$WC2026_FRONTEND_ORIGIN` (production custom domain).

## Deploy

Fly.io for the backend, Vercel for the frontend.
See [`docs/deploy.md`](docs/deploy.md) for the full runbook.

## Repo layout

```
src/wc2026/
  api/            FastAPI app + 11 routers (matches, predictions, tournament,
                  teams, h2h, ops, explain, live, track_record, health)
  ingest/         9 data-source loaders (Jürisoo Kaggle, eloratings.net,
                  football-data.org, TheSportsDB, openfootball, Wikipedia,
                  StatsBomb, FBref, football-data.co.uk, live_events poller)
  features/       time decay + match weights, host-team flags, rest days,
                  xG-form rolling, live-state replay, build_match_features
  models/         poisson_dc, shootout, elo_prior, xg_shot_model,
                  xgb_classifier, shap_explain, blend, live_win_prob
  eval/           backtest harness, calibration metrics, isotonic, rolling
  sim/            tournament Monte Carlo, groups, bracket, third_place,
                  knockout, conditional (Phase 8 known-results)
  db/             SQLAlchemy 2.0 models + Alembic
  scheduler/      APScheduler cron + interval jobs
  observability/  Sentry init + S3/R2 backup upload

frontend/
  src/app/        9 App Router routes
  src/components/ shadcn primitives + page-scoped chart components
  src/hooks/      useLiveWinProb (SSE), useLockedBracket (localStorage),
                  usePngDownload (html-to-image), useTeamAssets
  src/content/    methodology.mdx — synced from docs/methodology.md
  e2e/            Playwright smoke spec
  scripts/        sync-methodology.mjs (runs in prebuild)

alembic/          6 migrations covering 11 application tables
docs/             ARCHITECTURE.md, methodology.md, deploy.md, LICENSES.md
tests/            556 unit tests
data/             gitignored (raw parquet snapshots + artefacts)
scripts/          CLI entrypoints invoked by the scheduler
```

## Headline numbers

- **WC 2022 hindcast log-loss = 1.0379** — competitive with published bookmaker
  numbers (~0.95-1.00 closing-odds aggregate). Unchanged across every phase.
- **556 unit tests** (backend) + **26 Vitest** + **10 Playwright** (frontend).
- **13 cron jobs** + **3 interval-triggered tournament-window jobs** + **2 manual-only**.

## Licence

MIT for code. Data files retain upstream licences — see
[`docs/LICENSES.md`](docs/LICENSES.md).
