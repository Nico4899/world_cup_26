# Architecture

One Python repo. Three runtime processes (API, dashboard, scheduler) talk to one
PostgreSQL database. All data lives on disk or in Postgres; nothing in-process is
load-bearing for restart safety.

```mermaid
flowchart LR
    subgraph SRC[Free data sources]
        K[Kaggle: Jürisoo intl results]
        E[eloratings.net]
        F[football-data.org]
    end

    subgraph ING[Ingest layer]
        IK[kaggle_intl.py]
        IE[eloratings_scraper.py]
        IF[football_data_org.py]
    end

    subgraph DISK[On-disk artefacts]
        D1[(data/raw/jurisoo/*.csv)]
        D2[(data/raw/elo/*.parquet)]
    end

    subgraph DB[PostgreSQL 16]
        T1[(raw_matches)]
        T2[(raw_elo_snapshots)]
        T3[(model_predictions)]
        T4[(tournament_sim_runs)]
        T5[(scheduler_job_runs)]
    end

    subgraph ML[Model + simulator]
        M[PoissonDC + Dixon-Coles]
        S[Monte Carlo simulator]
    end

    subgraph SVC[Runtime services]
        SCH[APScheduler<br/>3 daily jobs]
        API[FastAPI<br/>6 endpoints]
        UI[Streamlit<br/>6 pages]
    end

    K --> IK --> D1
    E --> IE --> D2
    F --> IF
    IK --> T1
    IE --> T2
    SCH -.daily.-> IK & IE & IF
    SCH -.logs.-> T5

    D1 --> M
    D2 --> M
    M --> S
    M --> T3
    S --> T4
    API --> M
    API --> S
    UI --> API
```

## Process responsibilities

| Service | Module | Purpose | Restart safety |
|---|---|---|---|
| FastAPI | `src/wc2026/api/main.py` | Serves prediction + tournament endpoints; fits a PoissonDC model in the lifespan handler at startup | Stateless — re-fits on restart |
| Streamlit | `dashboard/streamlit_app.py` | Thin client over the API; cached via `@st.cache_data` (TTL 5 min) | Stateless — no DB writes |
| Scheduler | `src/wc2026/scheduler/jobs.py` | Three daily ingest jobs (04:00/04:15/04:30 UTC); logs runs to `scheduler_job_runs` | Re-registers cron triggers on startup; missed runs are simply skipped (no catch-up) |

## Data flow guarantees

- **Ingest is idempotent.** Re-running `download_kaggle_intl.py` overwrites the CSV; `scrape_eloratings.py` writes a fresh dated Parquet. No `INSERT` duplicates.
- **Model fit is deterministic.** Same input matches + same weights + same ref_date → same parameters (scipy.optimize with fixed seed-less L-BFGS-B).
- **Monte Carlo is seeded.** Same seed + same model → identical tournament. Cached in the API by `(n_sims, seed)`.
- **No live polling.** Dashboard reads cached predictions; no SSE, no WebSocket, no 60-second loops. Update cadence is whatever the scheduler runs.

## What is intentionally NOT here

See the "Out of scope" section in [README.md](../README.md) and the prescriptive review at `/Users/nico/.claude/plans/extensively-review-and-understand-iterative-fern.md`. Short list: no XGBoost ensemble, no xG features, no SHAP, no in-match win probability, no interactive bracket simulator, no Team Profile page, no Sentry, no PyDeck map.
