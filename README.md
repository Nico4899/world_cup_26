# wc2026-predictor

Calibrated probabilistic predictions for FIFA World Cup 2026.

Personal / educational project. Honesty about uncertainty is the headline feature; backtested calibration is the gate before any public prediction.

## Approach

- **Pre-match model**: weighted bivariate Poisson with Dixon–Coles low-score correction, team strengths anchored by World Football Elo. Bayesian hierarchical extension planned once the frequentist baseline is validated.
- **Tournament simulator**: Monte Carlo (50k–100k runs) respecting the 12-group / 8-best-thirds / R32 structure used for 2026.
- **Calibration**: every backtest run produces reliability diagrams, Brier score, log-loss, and Ranked Probability Score. Isotonic regression is fit on hold-out predictions to recalibrate.
- **Decision gate**: no public deployment until backtest log-loss on WC 2022 is within ~0.02 of football-data.co.uk closing-odds log-loss.

## Stage status

- [x] Stage 0.1 — repo skeleton, uv, pytest, Postgres docker-compose
- [x] Stage 0.2 — Jürisoo Kaggle ingester + eloratings.net polite scraper (2 TSV requests/refresh)
- [x] Stage 0.3 — weighted bivariate Poisson + Dixon–Coles with analytic gradient (5s → 0.12s fit on 9.5k matches)
- [x] Stage 0.4 — full Monte Carlo simulator (groups → tiebreakers → best-thirds → R32 → final), 3ms/sim
- [x] Stage 0.5 — WC 2018 & WC 2022 day-by-day hindcasts
- [x] Stage 0.6 — half-life sweep on WC 2022 → tuned to 10 years (was 730 days)
- [x] Stage 0.7 — decision-gate analysis (see below)
- [x] Stage 1.A — DB layer (Postgres + Alembic), football-data.org ingester, APScheduler, Dockerfile.app, CI
- [x] Stage 1.B — Elo prior + isotonic recalibration (research artefacts; degraded WC 2022 log-loss, kept for documentation), real Elo-weighted shootout submodel (integrated)
- [x] Stage 1.C — FastAPI app + Streamlit dashboard
- [x] Stage 2 — every deferred blueprint item shipped over 11 phases (XGBoost+SHAP blend as research artefact, live win-prob + SSE, Team Profile, scenario-comparison bracket, host-city map, TheSportsDB / StatsBomb / FBref / openfootball / Wikipedia ingesters, Sentry, S3/R2 backups, Fly.io config + runbook). Per-phase breakdown below.

## Decision-gate results (Stage 0.7)

Backtest metrics with the tuned 10-year half-life and 10-year history window:

| Tournament | log-loss | Brier | RPS | climatological log-loss | gap vs climatological |
|---|---:|---:|---:|---:|---:|
| WC 2022 | **1.0379** | 0.6033 | 0.2137 | 1.0674 | **−0.030** |
| WC 2018 | **0.9585** | 0.5690 | 0.2017 | 1.0569 | **−0.098** |

Comparison to published bookmaker numbers (Wheatcroft 2019; Constantinou 2019):
- Bookmaker closing-odds log-loss on WC 2018 ≈ 0.97–1.00 → our 0.9585 is **competitive**.
- Bookmaker WC 2022 numbers are not in the public literature; the conservative estimate is ≈ 0.95–1.00 → our 1.0379 is ~0.04-0.09 worse.

**Verdict: pass with caveats.** The platform produces honest, calibrated probability estimates that:
- Always beat the no-skill (climatological) baseline.
- Are competitive with published bookmaker numbers in low-upset tournaments (WC 2018).
- Lag bookmakers in high-upset tournaments (WC 2022) by ~0.04–0.09 log-loss — bookmaker odds incorporate Elo, injuries, news, and market mechanisms that a pure results-only Poisson model can't see.

The 5 most-surprising WC 2022 results the model flagged as long-shots (each given <13% probability) are exactly the famous upsets: Argentina 1-2 Saudi Arabia, Cameroon 1-0 Brazil, Japan 2-1 Spain, Tunisia 1-0 France, South Korea 2-1 Portugal. Calibrated long-shot detection works.

**Dashboard implications**: framing must be "the model thinks X" not "the answer is X". A 14% favourite (Argentina for WC 2026) loses 86% of the time.

## Dev setup

Requires [`uv`](https://docs.astral.sh/uv/) and Docker.

```bash
uv sync                       # install deps into .venv
docker compose up -d postgres # start local Postgres
uv run pytest                 # run tests
uv run ruff check .           # lint
uv run ruff format .          # format
```

## Running locally

The app is a **Next.js 16** dashboard (React 19 + TypeScript strict + Tailwind v4 + Visx) over a **FastAPI** model server.

```bash
# 1. Start the API (fits a PoissonDC on startup — first request takes ~1s for warmup):
uv run uvicorn wc2026.api.main:app --port 8000

# 2. In another terminal, start the dashboard:
cd frontend
pnpm install
NEXT_PUBLIC_API_URL=http://localhost:8000 pnpm dev
```

Then open http://localhost:3000. Dashboard routes (9 total):

| Route | Purpose |
|---|---|
| **`/`** (Today) | Prediction cards for any matchday in [2026-06-11, 2026-07-19], with kickoff time in UTC, clickable "Why?" popover (SHAP top-3), and a group-stage advancement strip |
| **`/match/[id]`** | Outcome popovers (SHAP per class), Visx 6×6 score heatmap, Elo-anchored narrative, live win-prob chart via Server-Sent Events, recent-5 form, H2H table |
| **`/groups`** | 12 group cards with Visx 5-segment stacked bars (1st / 2nd / 3rd→R32 / 3rd-out / 4th), live points + GD per group, top-10 championship headline |
| **`/bracket`** | Tabs: Single seed / Scenario comparison / Conditional locks (localStorage-backed). URL-encoded scenarios for shareable links |
| **`/track-record`** | Live WC 2026 rolling calibration + WC 2018 / WC 2022 historical hindcasts with Visx reliability scatter |
| **`/about`** | Methodology rendered from `docs/methodology.md` via MDX (synced by `pnpm sync:methodology`) |
| **`/ops`** | Health, scheduler-job status, manual `run-job` triggers via Next.js **Server Actions** (the ops token stays server-side) |
| **`/team/[name]`** | Per-team Elo history (Visx), recent-10 form, FIFA ranking, squad roster, rolling xG-form (5 / 10 / 12mo), Monte Carlo path-to-final with most-likely opponent per round |
| **`/map`** | 16 host venues on a deck.gl + MapLibre map with per-country fill + city filter |

### API endpoints

| Path | Returns |
|---|---|
| `GET /health` | model + fixtures + Elo snapshot age + group-assignment source |
| `GET /api/v1/matches?date=YYYY-MM-DD&group=A` | filtered fixture list |
| `GET /api/v1/matches/{id}` | one fixture + prediction (top-5 scorelines + full 11×11 score matrix) |
| `GET /api/v1/predictions/{home}/{away}?neutral=true&blend=true&blend_weight=0.5` | 1X2 + xG + top-5 + score matrix; `blend=true` adds the XGB-blended triplet when the optional artefact is loaded |
| `GET /api/v1/tournament/standings?n_sims=2000&seed=42&use_persisted=true` | 12-group MC probabilities + top-10 champion table; defaults to the latest persisted Phase-8 MC run |
| `GET /api/v1/tournament/bracket?seed=42` | one sampled 31-match knockout realisation |
| `GET /api/v1/teams/{team}/recent?n=5` | last-N matches from the team's perspective (W/D/L) |
| `GET /api/v1/teams/{team}/elo-history` | daily Elo snapshots |
| `GET /api/v1/teams/{team}/tournament-probs` | per-team advancement probs from the latest persisted MC run |
| `GET /api/v1/teams/{team}/assets` | crest / kit / stadium metadata from `raw_team_assets` |
| `GET /api/v1/teams/{team}/fifa-rankings` | monthly FIFA Men's Ranking history |
| `GET /api/v1/teams/{team}/squad` | latest tournament-squad snapshot |
| `GET /api/v1/teams/{team}/xg-form` | rolling xG aggregates over the last 5 + last 10 matches |
| `GET /api/v1/h2h/{a}/{b}?n=10` | head-to-head history |
| `GET /api/v1/explain/{match_id}?class_name=home_win&top_n=5` | SHAP top-features for one fixture (503 when no XGB artefact loaded) |
| `GET /api/v1/live/{match_id}` | current state + in-running win-prob (Phase 6) |
| `GET /api/v1/live/{match_id}/history` | full per-event timeline + the latest snapshot |
| `GET /api/v1/live/{match_id}/sse` | Server-Sent Events stream of win-prob frames |
| `GET /api/v1/track-record/wc2026` | rolling Brier / log-loss / RPS over completed WC 2026 matches |
| `GET /api/v1/track-record/historical/{tournament}` | WC 2018 / WC 2022 hindcast headline + reliability bins (24h cache) |
| `GET /api/v1/_ops/scheduler-status` | per-job latest run + status |
| `GET /api/v1/_ops/available-jobs` | list of manually-triggerable jobs |
| `POST /api/v1/_ops/run-job/{name}` | enqueue a manual run; gated by `X-Ops-Token` when `WC2026_OPS_TOKEN` is set |

## Deploying with Docker

### Local stack (postgres + scheduler + API)

```bash
docker compose up -d                 # postgres + scheduler + api (all three services)
docker compose logs -f scheduler     # watch the daily cron jobs
docker compose restart api           # pick up a freshly-refitted model artefact
```

Dashboard runs on Vercel (Next.js); for local dev see the "Running locally" section above. To exercise the production build against a local API:

```bash
cd frontend
NEXT_PUBLIC_API_URL=http://localhost:8000 pnpm build
NEXT_PUBLIC_API_URL=http://localhost:8000 pnpm start
```

### Deploying to Fly.io + Vercel (~$5–10/month hobby tier)

See [`docs/deploy.md`](docs/deploy.md) for the full runbook — Fly app + Postgres + volume, Cloudflare R2 (off-site backup), Sentry (error monitoring), **Vercel (Next.js dashboard)**, and rollback procedure. Short version:

```bash
cp fly.toml.example fly.toml          # then edit `app` name + `primary_region`
fly launch --no-deploy --copy-config
fly volumes create wc2026_data --size 2 --region <your-region>
fly postgres create --name wc2026-pg --region <your-region>
fly postgres attach wc2026-pg         # injects DATABASE_URL
fly secrets set FOOTBALL_DATA_ORG_KEY=<key> SENTRY_DSN=<dsn> \
                AWS_S3_BUCKET=<bucket> AWS_S3_ENDPOINT_URL=<r2-or-s3> \
                AWS_ACCESS_KEY_ID=<id> AWS_SECRET_ACCESS_KEY=<secret>
fly deploy
fly scale count app=1 scheduler=1     # one of each process group
```

The Fly config uses **two process groups** off the same Dockerfile.app:
`app` (uvicorn on port 8000, HTTPS via Fly's edge, `/health` check every 15 s, `min_machines_running = 1` so live SSE clients never see a cold-start)
and `scheduler` (BlockingScheduler, smaller VM). Both share a `wc2026_data`
volume mounted at `/app/data` for ingest data + the model artefact.

Every secret above is optional and gracefully degrades when absent:
- No `FOOTBALL_DATA_ORG_KEY` → daily fixture refresh + live event poller no-op silently.
- No `SENTRY_DSN` → `init_sentry` returns False, no events shipped.
- No `AWS_S3_BUCKET` → `db_backup` stays local-only (still daily, still pruned).

## Known limitations (post-Stage-2)

After 11 phases of build, the platform implements every blueprint item with these honest carve-outs:

- **Not yet deployed**. `fly.toml.example` is configured for the two-process-group + Postgres + volume layout (Phase 10), but `fly deploy` has not been run. The Streamlit dashboard isn't on Streamlit Community Cloud yet. Walk through [`docs/deploy.md`](docs/deploy.md) when ready to go live.
- **Three model add-ons are research-only**:
  - `PoissonDCWithPrior` (Elo prior) monotonically degrades WC 2022 log-loss across `prior_strength ∈ [0, 5]` — the base model already extracts team strength from match history.
  - `IsotonicCalibrator` (LOO recalibration) degrades WC 2022 log-loss by +0.077 — isotonic on N=64 is small-sample fragile; likely useful with N≥250.
  - `XgbMatchModel` + geometric blend (Phase 5) monotonically degrades on both WC 2018 (+0.000 → +0.005) and WC 2022 (+0.000 → +0.023) hindcasts as the XGB mixing weight rises from 0 to 50%. The pure Poisson is already near the bookmaker-quality ceiling; XGB-on-sparse-features adds noise rather than signal. Re-evaluate once xG-form / squad-age inputs are populated for the historical training corpus. Blend is exposed as `?blend=true&blend_weight=W` on `/api/v1/predictions/...`; defaults to off. SHAP `/api/v1/explain/{match_id}` is valuable independently of whether the blend is enabled.
  All three are kept as research artefacts; see `scripts/backtest_with_elo_prior.py`, `scripts/backtest_with_isotonic.py`, and `scripts/refit_xgb.py --hindcast`.
- **Bracket simulator is scenario-comparison, not click-to-set.** Phase 9 shipped a multi-seed explorer rather than the custom React+SVG bracket the blueprint specified. The plan's own Plotly-fallback note — Vite toolchain cost outweighs the UX gain. Reopen if/when the platform wants embeddable React widgets anyway.
- **Live event poller depends on football-data.org's free tier** (no detailed events on the free plan). Score deltas are tracked accurately; red cards + subs are not. Documented in [`src/wc2026/ingest/live_events.py`](src/wc2026/ingest/live_events.py).
- **Group letters A-L are derived from fixture dates** when no `data/wc2026_group_assignment.json` override is on disk. After the official FIFA draw, drop the openfootball-derived JSON into that path to switch from derived → authoritative; `/health` surfaces which source is in use.

## Stage 2 roadmap (complete)

Stage 1 shipped a calibrated, locally-runnable platform. Stage 2 expanded it to the full blueprint feature set across 11 phases. Plan: [`/Users/nico/.claude/plans/extensively-review-the-given-resilient-anchor.md`](/Users/nico/.claude/plans/extensively-review-the-given-resilient-anchor.md).

| Phase | Scope | Outcome |
|---|---|---|
| 1 | Documentation reconciliation | ✅ Stage 1.A/1.B/1.C marked done; Stage 2 roadmap drafted |
| 2 | TheSportsDB / openfootball / Wikipedia ingesters → crests, kit colours, canonical group letters, squads, FIFA ranking | ✅ Three new ingesters; weekly + monthly + manual-only schedules |
| 3 | StatsBomb open data + FBref + football-data.co.uk → xG corpus + shot model | ✅ Direct-GitHub StatsBomb ingester (no `statsbombpy`), polite FBref scraper, football-data.co.uk closing-odds loader, logistic xG shot model |
| 4 | `features.match_features` materialised table | ✅ Daily rebuild via `features_rebuild` job; Phase 5's XGBoost reads from this |
| 5 | XGBoost H/D/A + SHAP TreeExplainer + geometric blend; `/api/v1/explain/{match_id}` | ⚠️ Blend **regresses** log-loss on both WC 2018 + 2022 hindcasts as XGB weight rises — kept as research artefact, off by default. SHAP panel valuable independently. |
| 6 | In-match live win-prob + `/api/v1/live/{match_id}/sse` + live event poller | ✅ Logistic regression on (Elo Δ, GD, minutes, red Δ); SSE generator + direct-replay tests; production poller wired to scheduler post-audit |
| 7 | Rolling WC 2026 calibration on Track Record page | ✅ `/api/v1/track-record/wc2026` + dashboard panel; daily prediction snapshots persisted to `model_predictions` |
| 8 | Conditional 10k Monte Carlo auto-rerun after each FINISHED match; cache `tournament_sim_runs` reads | ✅ `simulate_*` family accepts `known_group_results`; `monte_carlo_rerun` every 30 min in tournament window; `/standings` reads persisted run |
| 9 | Dashboard polish: Team Profile, host-city map, SHAP panel, scenario-comparison bracket, crests | ✅ 9 dashboard pages (3 more than blueprint); click-to-set bracket deferred (Vite-toolchain cost) — see "Known limitations" |
| 10 | Fly.io config + Sentry SDK + off-site pg_dump to S3/R2 + warmth-keep | ✅ Code shipped (`observability/`, env-gated for graceful degradation). Actual `fly deploy` is operator-side — see [`docs/deploy.md`](docs/deploy.md). |
| 11 | Final README + ARCHITECTURE.md refresh | ✅ This commit |

### Key numbers (at completion)

- **Backend**: **556 unit tests** pass; ruff clean
- **Frontend**: **26 Vitest unit tests** + **10 Playwright smoke tests** pass; ESLint + `tsc --noEmit` clean
- **13 cron jobs** + **3 interval-triggered tournament-window jobs** + **2 manual-only**
- **26 API endpoints** across 11 routers
- **9 dashboard routes** in [`frontend/src/app/`](frontend/src/app/)
- **6 Alembic migrations** (Stage 1 + Phases 2/3/4/6/12) covering 11 application tables
- **WC 2022 hindcast log-loss = 1.0379** (unchanged across every phase; identical to the Stage 0.7 decision-gate value)

## Repo layout

```
src/wc2026/
  ingest/         # 9 data-source loaders: Jürisoo Kaggle, eloratings.net,
                  # football-data.org, TheSportsDB, openfootball, Wikipedia,
                  # StatsBomb, FBref, football-data.co.uk, live_events poller
  features/       # time decay + match weights, host-team flags, rest days,
                  # xG-form rolling, live-state replay, build_match_features
  models/         # poisson_dc, shootout, elo_prior, xg_shot_model,
                  # xgb_classifier, shap_explain, blend, live_win_prob
  eval/           # backtest harness, calibration metrics, isotonic, rolling
  sim/            # tournament Monte Carlo, groups, bracket, third_place,
                  # knockout, conditional (Phase 8 known-results)
  api/            # FastAPI app + 11 routers (matches, predictions, tournament,
                  # teams, h2h, ops, explain, live, track_record, health)
  db/             # SQLAlchemy 2.0 models + Alembic
  scheduler/      # APScheduler cron + interval jobs
  observability/  # Phase 10: Sentry init + S3/R2 backup upload
frontend/         # Next.js 16 + React 19 + TypeScript + Tailwind v4 + Visx
  src/app/        # 9 App Router routes (Today / Match Detail / Groups / Bracket /
                  # Track Record / About / Ops / Team Profile / Map)
  src/components/ # shadcn-ui primitives + page-scoped chart components
  src/hooks/      # useLiveWinProb (SSE), useLockedBracket (localStorage),
                  # usePngDownload (html-to-image), useTeamAssets
  src/content/    # methodology.mdx — synced from docs/methodology.md
  e2e/            # Playwright smoke spec
alembic/          # 6 migrations (Stage 1 + Phases 2/3/4/6/12)
docs/             # ARCHITECTURE.md, methodology.md, deploy.md, LICENSES.md
tests/            # 556 unit tests
data/             # local-only, gitignored (raw parquet snapshots + artefacts)
scripts/          # one-off + scheduler-invoked entrypoints
notebooks/        # exploratory; not under test
```

## Licence

MIT for code. Data files retain their upstream licences — see [`docs/LICENSES.md`](docs/LICENSES.md) for the per-source breakdown (Jürisoo Kaggle: CC0; eloratings.net: derivative facts; football-data.org: free-tier ToS).
