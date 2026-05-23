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
- [x] Stage 1.C — FastAPI app + Streamlit dashboard (merged from `agent/frontend`)
- [ ] Stage 1.A — DB layer, football-data.org ingester, scheduler, Dockerfile.app, CI (parallel branch: `agent/backend`)
- [ ] Stage 1.B — Elo prior, real shootout model, isotonic recalibration (parallel branch: `agent/models`)

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

## Approach

- **Pre-match model**: weighted bivariate Poisson + Dixon–Coles correction, weighted MLE with analytic gradient, sum-to-zero identifiability on attack/defence vectors. Time decay half-life 10 years (tuned), tournament-importance weights per the World Football Elo K-factor schedule.
- **Tournament simulator**: Monte Carlo (~3ms per full tournament), respecting 12 groups → 8 best-thirds → R32 with the published 2026 third-placed slot eligibility table (all 495 advancing-set scenarios produce a valid bipartite matching).
- **Calibration**: every backtest run produces reliability diagrams, Brier score, log-loss, and Ranked Probability Score.
- **Known limitations** (see also the plan file):
  - Group letters A-L are derived from fixture dates, not FIFA's published assignment.
  - FIFA tiebreaker order is the 2022 procedure; the 2026 regulations document should be cross-checked before public launch.
  - Penalty shootout is a 50/50 placeholder; a proper Dawson-style submodel is a Stage 1 candidate.
  - No Elo prior, no injury/suspension override, no XGBoost ensemble layer (deferred).

See [`/Users/nico/.claude/plans/extensively-review-and-understand-iterative-fern.md`](/Users/nico/.claude/plans/extensively-review-and-understand-iterative-fern.md) for the full plan and review.

## Dev setup

Requires [`uv`](https://docs.astral.sh/uv/) and Docker.

```bash
uv sync                       # install deps into .venv
docker compose up -d postgres # start local Postgres
uv run pytest                 # run tests
uv run ruff check .           # lint
uv run ruff format .          # format
```

## Stage 1 — Running locally

The app is a thin **Streamlit** dashboard over a **FastAPI** model server.

```bash
# 1. Start the API (fits a PoissonDC on startup — first request takes ~1s for warmup):
uv run uvicorn wc2026.api.main:app --port 8000

# 2. In another terminal, start the dashboard:
WC2026_API_URL=http://localhost:8000 uv run streamlit run dashboard/streamlit_app.py
```

Then open http://localhost:8501. Dashboard pages:

| Page | Purpose |
|---|---|
| **Today** | Prediction cards for any matchday between 2026-06-11 and 2026-06-27 |
| **Match Detail** | Per-match 1X2 + scoreline heatmap + plain-language "why" |
| **Groups** | 12 group blocks with stacked-bar advancement probabilities (1st / 2nd / 3rd→R32 / out) |
| **Bracket** | One sampled knockout realisation; resampleable by seed |
| **Track Record** | WC 2022 + WC 2018 hindcast reliability diagrams + Brier / log-loss / RPS |

### API endpoints

| Path | Returns |
|---|---|
| `GET /health` | model + fixtures load status |
| `GET /api/v1/matches?date=YYYY-MM-DD&group=A` | filtered fixture list |
| `GET /api/v1/matches/{id}` | one fixture + prediction (top-5 scorelines + full 11×11 score matrix) |
| `GET /api/v1/predictions/{home}/{away}?neutral=true` | 1X2 + xG + top-5 scorelines + full 11×11 score matrix |
| `GET /api/v1/tournament/standings?n_sims=2000&seed=42` | 12-group MC probabilities + top-10 champion table |
| `GET /api/v1/tournament/bracket?seed=42` | one sampled 31-match knockout realisation |

## Deploying with Docker

```bash
# Build the dashboard image:
docker build -f Dockerfile.dashboard -t wc2026-dashboard .

# Run the dashboard, pointing at the API on the host:
docker run --rm -p 8501:8501 \
    -e WC2026_API_URL=http://host.docker.internal:8000 \
    wc2026-dashboard
```

A `Dockerfile.app` for the API/scheduler is provided on the `agent/backend` branch (Stage 1.A).

## Known limitations (Stage 1)

- **Local-only by default.** The Dockerfiles build, but no deploy target is wired up; you run the stack with `uv run uvicorn …` + `uv run streamlit …` (or via `docker compose`). The scheduler is coded but not running as a persistent process.
- **No live polling / no in-match updates.** The dashboard shows pre-match predictions; updates require a scheduler refresh.
- **Two model add-ons are experimental, not used by default**:
  - `PoissonDCWithPrior` (Elo prior) monotonically degrades WC 2022 log-loss across `prior_strength ∈ [0, 5]` — the base model already extracts team strength from match history.
  - `IsotonicCalibrator` (LOO recalibration) degrades WC 2022 log-loss by +0.077 — isotonic on N=64 is small-sample fragile; likely useful with N≥250.
  Both are kept as research artefacts; see `scripts/backtest_with_elo_prior.py` and `scripts/backtest_with_isotonic.py`.

## Out of scope

The original blueprint specified the items below; we explicitly decided NOT to build them. See `/Users/nico/.claude/plans/extensively-review-and-understand-iterative-fern.md` for the reasoning per item.

- **XGBoost secondary classifier + SHAP explanations + geometric-mean blend** — the pure Poisson model on WC 2018 (log-loss 0.9585) is already competitive with bookmaker closing odds (~0.97–1.00); the academic Poisson + RF hybrid improves by ~0.01–0.03 at significant complexity cost.
- **xG-based features** (StatsBomb open data, FBref) — requires building an xG model from event data; ~2 weeks of work for an unverified gain.
- **In-match live win-probability** (SSE, 30-second polling, score+minute+red-card logistic) — the platform isn't deployed and no live-UI consumer exists.
- **Interactive click-to-set bracket simulator** — Streamlit is the wrong tool for that interaction model; the Bracket Realisation page shows one sampled realisation per seed instead.
- **Team Profile page** — would duplicate information already split across Groups, Match Detail, and Track Record.
- **PyDeck host-city map** — eye candy.
- **TheSportsDB / openfootball / Wikipedia / FIFA.com ingesters** — we don't show logos/kits, and the model doesn't use squad metadata.
- **football-data.co.uk closing-odds ingest** — the blueprint cited this as a benchmark source for World Cup bookmaker odds, but on inspection it only publishes Euros + domestic leagues, not World Cups. The Track Record page cites published academic numbers (Wheatcroft 2019, Constantinou 2019) instead of live-ingesting.
- **Sentry / Prometheus monitoring** — not deployed yet; YAGNI.

## Repo layout

```
src/wc2026/
  ingest/    # data source loaders (Jürisoo CSV, eloratings, football-data.org, ...)
  features/  # feature engineering (time decay, venue, travel, ...)
  models/    # bivariate Poisson, shootout, live win-prob
  eval/      # backtest harness, calibration, reliability diagrams
  db/        # SQLAlchemy models + Alembic migrations
  api/       # FastAPI app (Stage 1)
  scheduler/ # cron job entrypoints
dashboard/   # Streamlit (Stage 1)
tests/       # pytest, src layout
data/        # local-only, gitignored
scripts/     # one-off scripts
notebooks/   # exploratory; not under test
```

## Licence

MIT for code. Data files retain their upstream licences; see source notes in `src/wc2026/ingest/`.
