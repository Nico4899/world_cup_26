# Agent guide

Read this before editing. See [README.md](README.md) for project shape and
quickstart.

## Project layout (at a glance)

- **`src/wc2026/`** — Python backend (FastAPI + APScheduler + SQLAlchemy 2.0).
  See `src/wc2026/api/` for the 11 routers and `src/wc2026/sim/` for the
  Monte Carlo simulator.
- **`frontend/`** — Next.js 16 (App Router) + React 19 + TypeScript strict +
  Tailwind v4 + Visx + shadcn/ui on top of `@base-ui/react`. Vercel target.
- **`tests/`** — pytest. `tests/unit/` runs by default; mark slow/network
  tests with `@pytest.mark.slow` or `@pytest.mark.integration`.
- **`frontend/e2e/`** — Playwright. **`frontend/src/**/*.test.{ts,tsx}`** — Vitest.
- **`alembic/`** — DB migrations. New schema change = new migration file.
- **`docs/`** — `ARCHITECTURE.md` (as-built), `methodology.md` (model docs;
  authoritative source for `frontend/src/content/methodology.mdx`),
  `deploy.md`, `LICENSES.md`.
- **`scripts/`** — CLI entrypoints invoked by the scheduler.

## Common commands

```bash
# Backend
uv sync                          # install deps
uv run pytest                    # unit tests (~20s, 556 tests)
uv run pytest -m slow            # slow tests (opt in)
uv run ruff check .              # lint
uv run ruff format .             # format
uv run alembic upgrade head      # apply migrations
uv run uvicorn wc2026.api.main:app --port 8000  # dev server

# Frontend (cwd: frontend/)
pnpm install
pnpm dev                         # Next.js on :3000
pnpm typecheck                   # tsc --noEmit
pnpm lint                        # eslint
pnpm test:run                    # Vitest, 26 tests
pnpm e2e                         # Playwright smoke
pnpm build                       # production build
pnpm sync:methodology            # regenerate src/content/methodology.mdx
pnpm gen:types                   # regenerate src/lib/api-types.ts from OpenAPI
```

## Code style

**Python**
- Python 3.12+, `ruff` for lint + format (config in `pyproject.toml`;
  `[tool.ruff.lint] select` includes E/F/W/I/B/UP/SIM/RUF/PL/NPY/PD).
- Type hints required. Pydantic v2 models for API request/response schemas
  live in `src/wc2026/api/schemas.py`.
- Imports: `from __future__ import annotations` at the top of every module.
- One module = one cohesive concept. Helpers shared across two callers go
  into a parent module, not a `utils.py` grab-bag.
- Avoid `print()`; use `logging.getLogger(__name__)`.

**TypeScript / React**
- `tsconfig.json` is strict. Don't relax it.
- App Router conventions: default to **Server Components**; mark
  `"use client"` only where you need interactivity (forms, popovers, SSE,
  TanStack Query hooks).
- Data fetching: server-side `apiGet` from `@/lib/api`; client-side
  TanStack Query. URL state via `useSearchParams` + `router.replace`.
  `localStorage` only via the existing `useLockedBracket` pattern.
- Charts: **Visx**. Replace Plotly-style imports with Visx equivalents.
- UI primitives: shadcn (`@/components/ui/*`). Don't add components that
  duplicate existing primitives.
- Tailwind: prefer utility classes; canonical class warnings from the IDE
  (e.g. `w-[380px]` → `w-95`) should be fixed.

## Tests are gates

- The **WC 2022 hindcast log-loss must stay at 1.0379**. If a change moves
  it, the hindcast scripts (`scripts/backtest_wc2022.py`) will surface the
  drift. Don't merge changes that regress this number.
- New API endpoints need a unit test in `tests/unit/test_api_routes.py`.
- New components: add a Vitest test for pure logic + extend the Playwright
  smoke if the route added a new heading.

## Things to avoid

- **No new top-level directories** unless we genuinely need one. The repo
  layout is documented in [README.md](README.md); deviations bit-rot quickly.
- **No `node_modules` at the repo root.** The frontend has its own
  `frontend/node_modules`. Running `pnpm` from the repo root creates a
  stray (it's gitignored, but still wasteful).
- **No dependencies without justification.** Audit candidates locally
  before adding: every Python dep is grepped for imports in
  `src/`/`scripts/`/`tests/`; every Node dep is grepped in `src/` + configs.
  Unused deps land in PR review.
- **Don't regenerate `frontend/src/content/methodology.mdx` by hand** —
  edit `docs/methodology.md` and run `pnpm sync:methodology`.
- **Don't commit `.next/`, `node_modules/`, `data/raw/`, or local secrets**
  (`.env`, `.env.local`). The root `.gitignore` covers all of these.

## Authorisation hierarchy

- **Read-only / local edits / tests**: just do it.
- **Destructive ops** (git reset --hard, force push, deleting branches,
  dropping DB tables, removing tracked files): confirm with the user first
  even if the plan implies it.
- **Network calls outside localhost** (Fly, Vercel, GitHub PRs, Slack):
  only when explicitly asked.

## Reference

- [README.md](README.md) — quickstart, routes, deploy
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — as-built architecture
- [docs/methodology.md](docs/methodology.md) — model methodology + citations
- [docs/deploy.md](docs/deploy.md) — Fly + Vercel runbook
