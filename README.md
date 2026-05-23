# wc2026-predictor

Calibrated probabilistic predictions for FIFA World Cup 2026.

Personal / educational project. Honesty about uncertainty is the headline feature; backtested calibration is the gate before any public prediction.

## Approach

- **Pre-match model**: weighted bivariate Poisson with Dixon–Coles low-score correction, team strengths anchored by World Football Elo. Bayesian hierarchical extension planned once the frequentist baseline is validated.
- **Tournament simulator**: Monte Carlo (50k–100k runs) respecting the 12-group / 8-best-thirds / R32 structure used for 2026.
- **Calibration**: every backtest run produces reliability diagrams, Brier score, log-loss, and Ranked Probability Score. Isotonic regression is fit on hold-out predictions to recalibrate.
- **Decision gate**: no public deployment until backtest log-loss on WC 2022 is within ~0.02 of football-data.co.uk closing-odds log-loss.

## Stage status

- [x] Stage 0.1 — repo skeleton, deps, pytest, Postgres
- [ ] Stage 0.2 — ingest Jürisoo intl results CSV, scrape eloratings.net
- [ ] Stage 0.3 — weighted bivariate Poisson + Dixon–Coles fit
- [ ] Stage 0.4 — Monte Carlo tournament simulator + correctness tests
- [ ] Stage 0.5 — WC 2018 / WC 2022 / Euros 2020 / Euros 2024 hindcast
- [ ] Stage 0.6 — calibration sweep + isotonic recalibration
- [ ] Stage 0.7 — decision gate

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
