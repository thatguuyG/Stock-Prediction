# ADR 0011 — FastAPI thin shim over ORM-direct or PostgREST

**Status:** Accepted (2026-05-15)

## Context

The Phase 3.5 dashboard needs a way to read Postgres tables (positions, signals, orders, risk_state). Three architectural options:

1. **FastAPI shim** — write a few Python endpoints that the dashboard talks to.
2. **Dashboard talks to Postgres directly** via something like `node-postgres`, with its own connection string.
3. **PostgREST** — drop in an auto-generated REST API over Postgres and let the dashboard hit it.

## Decision

A thin FastAPI shim at [services/api/](../../services/api/). One endpoint per dashboard concern, reusing the existing SQLAlchemy ORM and `session_scope()` for transactional reads.

## Consequences

**Positive**
- Reuses the existing `packages/shared/{models,db,config}` — no duplicated schema definitions, no second connection-string sprawl, no second migration story.
- Pydantic response models give us auto-generated OpenAPI docs at `/docs`. Operators can poke endpoints from a browser without writing JS.
- Server-side aggregations (joins across signals + orders, derived `current_position_value` columns) can be added in Python without touching the dashboard.
- Tests reuse the existing SQLite-in-memory `session` fixture via FastAPI's `dependency_overrides`. We pay nothing extra for test infrastructure.
- The shim is the natural surface for any future write endpoints (manual overrides, Phase 4) — we won't have to introduce a second framework when that day comes.

**Negative**
- One more service to deploy when Phase 4 lands. PostgREST would have been a single container.
- Hand-writing endpoints means each new column the dashboard wants involves at least two diffs (ORM/Pydantic + router).
- Some duplication: the TypeScript types in [apps/dashboard/src/types/api.ts](../../apps/dashboard/src/types/api.ts) are hand-mirrored from Python Pydantic schemas. Auto-generation via `openapi-typescript` is a follow-up.

## Alternatives considered

- **Dashboard talks to Postgres directly.** Rejected on principle: the JS layer would need its own SQL knowledge, its own connection pooling, and we'd lose the ability to enforce business-level constraints (e.g. "only return positions where source = 'alpaca'") in one place. Also bad for any future deployment where the dashboard runs in an environment that shouldn't have Postgres credentials.
- **PostgREST.** Genuinely tempting — it would have given us a zero-Python REST layer over our schema for free. Rejected because (a) it can't easily reshape responses (no Pydantic-style projection), (b) it forces us to think about auth and row-level security in Postgres rather than in application code, and (c) it would introduce a second runtime we'd need to learn and operate.
- **GraphQL via Strawberry.** Rejected as overkill — we have four endpoints and zero need for query composition flexibility on the dashboard side.

## Revisit if

- The shim grows past ~20 endpoints and most are pure "select * from X with filters" — at that point an auto-generated layer becomes more attractive.
- We need to support a second client (mobile app, CLI, third-party integration) where REST-by-convention would save us shipping a new endpoint per use case.
- Latency becomes meaningful — FastAPI's Python overhead is fine at 4–5 endpoints/page but if the dashboard does dozens of round-trips PostgREST's lighter footprint would matter.
