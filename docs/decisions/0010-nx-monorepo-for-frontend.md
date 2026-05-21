# ADR 0010 — Nx for the Node side; Python stays Python

**Status:** Accepted (2026-05-15)

## Context

Phase 3.5 introduces the first Node code to the repo (a Next.js dashboard + a FastAPI shim that feeds it). The user has explicitly asked for an **Nx monorepo** so the new frontend and the existing Python backend live together. There were two coupling options:

1. **Full Nx managing both sides** via a community plugin like `@nxlv/python`. Nx targets would invoke Poetry/uv for Python, generators would scaffold Python projects, the dependency graph would span both languages.
2. **Nx for the Node side only.** Python stays at the repo root, owned by `pyproject.toml`. Nx provides a thin `run-commands` wrapper for the FastAPI service so `nx run-many` can boot both.

## Decision

Option 2. Nx is the workspace manager for the Node frontend; Python continues to be managed by `pyproject.toml` + `pip`. The FastAPI service has an [Nx project descriptor](../../services/api/project.json) that wraps `uvicorn` via `nx:run-commands`, which is enough to let `nx run-many --target=serve --projects=dashboard,api` start the whole dev stack.

## Consequences

**Positive**
- Each language is managed by its native tool. Python developers don't need to learn Nx; Node developers don't need to learn pip.
- No third-party Nx plugin to track. `@nxlv/python` is community-maintained and has historically lagged the Poetry/Nx ecosystem; we avoid that dependency.
- The wrapper is tiny — a 12-line `project.json` per Python service.
- `npm install` only touches Node deps; `pip install -e ".[dev]"` only touches Python deps. Each side can be rebuilt or wiped independently.

**Negative**
- Nx's dependency graph doesn't know about Python imports. If a Python module renames a function, Nx won't know to invalidate caches for the FastAPI service. We rely on Python's own pytest/pylint caches.
- Operators need to install both Node and Python toolchains to develop the full stack. The bar is "Node 20+ and Python 3.12+" — not onerous, but two ecosystems.
- Nx generators (`nx g @nx/next:app`) don't apply to our Python side; for new Python services we hand-write the `project.json`.

## Alternatives considered

- **`@nxlv/python`** would give Nx full visibility into Python projects (including caching). Rejected because it adds a non-trivial dependency, requires switching Python's package manager (typically Poetry), and would force us to maintain compatibility between the plugin and our pip-based workflow. The integration benefit doesn't justify the operational cost.
- **No Nx at all** — just `npm` in `apps/dashboard/` and Python at root. Rejected because the user explicitly asked for an Nx monorepo to coordinate the two sides. Nx also gives us a sensible `nx run-many --parallel` for dev orchestration.
- **`turbo` instead of Nx** — equally valid for Node-only orchestration, lighter footprint. Not chosen because the user asked for Nx specifically.

## Revisit if

- A second Python service grows complex enough that the cost of Nx not seeing its dependencies (cache misses, no impact analysis) becomes painful.
- Node CI runs in a way that meaningfully benefits from Nx's affected-graph feature (`nx affected -t test`) AND we'd want the Python side included in that calculation.
- The community plugin landscape stabilises around a clear winner for Python-in-Nx integration.
