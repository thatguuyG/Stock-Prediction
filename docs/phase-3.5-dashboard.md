# Phase 3.5 — Dashboard

> **Status: DEFERRED — not yet implemented.**
> This is a carve-out from Phase 3. The signal engine and broker landed without a UI; this phase adds one.

## Why this is its own phase

Phase 3 ships headless: positions, signals, and orders all reachable via `stockpred report` or `psql`. That's enough for development and for a hands-off paper-trading loop. A dashboard is a separate concern — separate language toolchain (Node + JS), separate deployment target, separate testing model — so it gets its own phase rather than padding Phase 3.

## Scope (when this lands)

- A FastAPI shim ([services/api/](../services/api/)) exposing the existing DB tables as JSON:
  - `GET /positions` — current portfolio from the `positions` table.
  - `GET /signals?limit=50&decision=BUY` — recent decisions with the rule trace.
  - `GET /orders?status=filled&limit=50` — order history.
  - `GET /equity-curve?from=...&to=...` — `risk_state` time series.
- A Next.js dashboard ([services/dashboard/](../services/dashboard/)) consuming the shim:
  - Positions panel + P&L.
  - Recent signals table with rationale (why each gate fired).
  - Equity curve chart.
  - Latest news headlines + sentiment (read-only join over `news_items` + `sentiments`).
- Read-only first. No write endpoints in Phase 3.5 — order submission stays inside cron jobs. Manual overrides are a Phase 4 concern.

## Open questions when this starts

- Auth: a single shared password? OAuth via GitHub? None (localhost-only)?
- Deployment: same Cloud Run as the other services, or static export to GitHub Pages + an API on Cloud Run?
- Real-time: do we want WebSocket-pushed updates, or is a 30-second polling refresh fine for a paper-trading dashboard?
- Component library: Tailwind + headless components, shadcn/ui, or full-bake like Mantine?

## Acceptance

- Loads in under 1 second on a fresh page hit.
- Renders the same data `stockpred report` does, with charts.
- No write endpoints; the dashboard is a window, not a controller.
- E2E test: spin up the API shim against the SQLite test DB and assert the dashboard's positions endpoint matches what `Position.qty` says.

## Predecessor / successor

- **Depends on:** [Phase 3](phase-3-execution.md) (data already flowing into `signals`, `orders`, `positions`, `risk_state`).
- **Successor:** Phase 4's monitoring stack can reuse the same FastAPI shim for ops dashboards.
