# ADR 0001 — Monorepo layout

**Status:** Accepted (2026-05-15)

## Context

We are growing a single-file LSTM script into a multi-component system: ingestion, model training, signal generation, broker integration, dashboard. Each will eventually have its own deploy unit. We need to decide between one repo for everything or one repo per service from day one.

## Decision

Keep everything in this repository, organised as:

```
services/<name>/     # deployable unit, ingestion/, model/, signal/, broker/, dashboard/
packages/shared/     # config, DB, ORM, logging shared across services
tests/               # all tests in one tree
migrations/          # one Alembic history for the whole DB schema
docs/                # this folder
```

Each service is a Python module today. When a service goes to production, it gets a `Dockerfile` next to its code. The dashboard, when it lands, will be a Next.js sub-folder.

## Consequences

**Positive**
- One CI run validates the whole system end-to-end.
- Refactors that span layers (rename a column, change a contract) happen in a single PR.
- Single dependency manifest (`pyproject.toml`) keeps versions aligned.
- Single Postgres schema with one Alembic history — no risk of divergent migrations.

**Negative**
- CI runs every test on every push regardless of which service changed. Mitigated for now by a small test suite; we'll add path-based job filters when CI time becomes a problem.
- All services share Python version + lockfile. Acceptable while everything is Python; will revisit when the Node dashboard arrives (likely a `package.json` sibling, not a separate repo).

## Alternatives considered

- **One repo per service**: cleaner blast radius but enormous boilerplate cost for a project this small. Cross-cutting refactors require coordinated PRs. Rejected.
- **Single flat Python package**: would work today but doesn't scale to the Node dashboard or future Rust/Go components. Rejected as a dead-end.

## Revisit if

- CI runtime exceeds ~10 min and path filters can't fix it.
- A service needs an incompatible runtime that complicates the shared environment (e.g. a model service pinned to Python 3.10 while the rest moves to 3.13).
