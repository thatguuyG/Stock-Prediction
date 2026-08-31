# Phase 3.5 — Dashboard

> **Status: implemented.** Read-only Next.js dashboard fed by a FastAPI shim. Operators can now eyeball positions, signals, and the equity curve in a browser.

## What's in

- **FastAPI shim** at [services/api/](../services/api/) exposing four GET endpoints plus a `/healthz` probe over the Phase 3 schema.
- **Next.js 16 dashboard** at [apps/dashboard/](../apps/dashboard/) with three pages: `/` (positions + summary), `/signals` (decisions with rationale), `/equity` (Recharts line chart).
- **Nx workspace** at the repo root managing the Node side. Python lives at root, unchanged; an Nx project wrapper at [services/api/project.json](../services/api/project.json) makes `nx run-many --target=serve` boot both services.
- **Localhost-only.** API binds to `127.0.0.1:8000`, dashboard to `localhost:3000`, dashboard rewrites `/api/*` to the FastAPI shim. No auth.

## Architecture

```
  ┌─────────────────────────────┐    rewrites /api/*    ┌────────────────────────┐
  │ apps/dashboard (Next.js 16) │ ───────────────────▶ │ services/api (FastAPI) │
  │ http://localhost:3000        │                       │ http://127.0.0.1:8000  │
  └─────────────────────────────┘                       └───────────┬────────────┘
                                                                    │ session_scope()
                                                                    ▼
                                                            ┌────────────────┐
                                                            │   Postgres     │
                                                            │ (or SQLite     │
                                                            │  in tests)     │
                                                            └────────────────┘
```

The FastAPI shim is **read-only** — no write endpoints. Order submission stays in cron jobs (`stockpred run-signals`); the dashboard is a window, not a controller. Manual overrides are deliberately deferred to Phase 4.

## CLI / dev workflow

```bash
# install Node deps (one time)
npm install

# in one terminal: API
nx serve api
# in another: dashboard
nx serve dashboard

# or both at once
npm run dev
# (alias for `nx run-many --target=serve --projects=dashboard,api --parallel`)
```

Production build of the dashboard:

```bash
nx build dashboard
```

Lint and typecheck (both run in CI as the `dashboard` job):

```bash
nx lint dashboard        # eslint flat config at apps/dashboard/eslint.config.mjs
nx typecheck dashboard   # tsc --noEmit
```

Next 16 removed the `next lint` command, so linting invokes the eslint CLI directly against
`eslint-config-next/core-web-vitals` + `eslint-config-next/typescript` in flat-config form.

## API endpoints

All return JSON. Implemented at [services/api/routers/](../services/api/routers/) and modelled in [services/api/schemas.py](../services/api/schemas.py).

| Method | Path | Query params | Response |
|---|---|---|---|
| GET | `/healthz` | — | `{"status": "ok"}` |
| GET | `/positions` | — | `list[Position]` (non-zero only) |
| GET | `/signals` | `limit` (1–500), `decision` (BUY/SELL/HOLD) | `list[Signal]`, newest first |
| GET | `/orders` | `limit`, `status` | `list[Order]`, newest first |
| GET | `/equity` | `from`, `to` (ISO dates) | `list[EquityPoint]`, ascending |

OpenAPI docs at `http://127.0.0.1:8000/docs` while the API is running.

## Dashboard pages

| Route | Source | What it shows |
|---|---|---|
| `/` | [src/app/page.tsx](../apps/dashboard/src/app/page.tsx) | Latest `risk_state` summary card + positions table |
| `/signals` | [src/app/signals/page.tsx](../apps/dashboard/src/app/signals/page.tsx) | Last 50 signals; click "rationale" to expand the JSON audit |
| `/equity` | [src/app/equity/page.tsx](../apps/dashboard/src/app/equity/page.tsx) | Recharts line chart of equity over time |

Server components by default; only the rationale toggle (`SignalsTable`) and Recharts (`EquityChart`) are client components.

### Fetching: why the API base URL is environment-dependent

[src/lib/api.ts](../apps/dashboard/src/lib/api.ts) picks its base URL by execution context:

```ts
const BASE =
  typeof window === 'undefined'
    ? `http://${process.env.API_HOST ?? '127.0.0.1'}:${process.env.API_PORT ?? '8000'}`
    : '/api';
```

The `/api/*` rewrite in [next.config.mjs](../apps/dashboard/next.config.mjs) only
applies to requests that reach the Next.js HTTP server — i.e. from a **browser**.
All three pages are async Server Components, so their fetches run in Node during
SSR, where a relative URL has no origin and `fetch` throws
`TypeError: Failed to parse URL from /api/positions`.

**Keep the server branch absolute.** Reverting it to a bare `/api` makes every
page render its empty state, because each one wraps its call in
`.catch(() => [])` — so a total API failure is visually identical to an empty
database. If the dashboard ever looks blank, check
`curl 127.0.0.1:8000/healthz` before suspecting the data pipeline.

Client-side calls keep `/api` so they continue to flow through the rewrite and
avoid CORS.

## Tech stack

| Layer | Choice | ADR |
|---|---|---|
| Monorepo | Nx 20 (Node side); Python stays Python | [0010](decisions/0010-nx-monorepo-for-frontend.md) |
| API framework | FastAPI + uvicorn | [0011](decisions/0011-fastapi-thin-shim-over-orm-direct.md) |
| Frontend | Next.js 16 App Router | User-chosen |
| Styling | Tailwind CSS v4 | Next.js default. v4 is configured in CSS (`@import "tailwindcss"` + `@theme` in [globals.css](../apps/dashboard/src/app/globals.css)) via `@tailwindcss/postcss` — there is no `tailwind.config.ts`. |
| Charts | Recharts | User-chosen |
| TS types | Hand-rolled in `src/types/api.ts` mirroring `services/api/schemas.py` | OpenAPI codegen is a future nicety |
| Auth | None — localhost only | User-chosen |

## Tests

| File | Coverage |
|---|---|
| [tests/test_api_health.py](../tests/test_api_health.py) | `/healthz` returns 200 |
| [tests/test_api_positions.py](../tests/test_api_positions.py) | Non-zero filter, empty-DB response, response shape |
| [tests/test_api_signals.py](../tests/test_api_signals.py) | Most-recent-first ordering, `limit`, `decision` filter, rationale roundtrip, 422 on bad enum |
| [tests/test_api_orders.py](../tests/test_api_orders.py) | Most-recent-first, status filter, limit cap, empty response |
| [tests/test_api_equity.py](../tests/test_api_equity.py) | Ascending order, `from`/`to` filter, response shape |

All FastAPI tests use `TestClient` with the `get_session` dependency overridden to use the existing SQLite-in-memory `session` fixture. **`conftest.py` was updated to use `StaticPool` + `check_same_thread=False`** so a single in-memory DB is shared across the FastAPI threadpool — see the conftest note.

No Node tests in this phase. Dashboard verified manually via `nx serve dashboard`.

## CI

Existing `lint.yml` + `test.yml` workflows still cover the Python side. Node CI is deliberately deferred until/unless there's a regression risk worth gating.

## Verification recipe

```bash
# Python
pip install -e ".[dev]"
pytest -q                                  # 91 tests pass (74 prior + 17 API)
pylint $(find services packages -name '*.py') tests/*.py   # 10.00/10

# Node
npm install
nx build dashboard                         # production build succeeds

# End-to-end smoke
nx serve api                               # term 1
nx serve dashboard                         # term 2
curl http://127.0.0.1:8000/healthz         # → {"status":"ok"}
open http://localhost:3000                  # browse the three pages
```

## What's still out of scope

- Production deployment (Phase 4).
- Write endpoints / manual overrides (Phase 4).
- Auth (added when we deploy publicly).
- Real-time push (WebSocket). The dashboard re-fetches on navigation; if you need live updates, refresh.
- E2E tests for the dashboard. Phase 4 can add a Playwright suite when the deployment story takes shape.

## Predecessor / successor

- **Depends on:** [Phase 3](phase-3-execution.md) (positions / signals / orders / risk_state populated).
- **Unlocks:** [Phase 4](phase-4-live-and-ops.md) — same FastAPI shim can be deployed and add monitoring panels.
