# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env       # then fill NEWSAPI_KEY for live news ingestion
```

The repo uses `pyproject.toml` (no Poetry). `uv venv .venv && uv pip install -e ".[dev]"` works if Python's stdlib `venv` is unavailable.

## Common commands

| Task | Command |
|---|---|
| Run full test suite | `pytest -q` |
| Run a single test | `pytest tests/test_features.py::test_rsi_in_valid_range` |
| Run pylint | `pylint $(find services packages -name '*.py') tests/*.py` |
| Run the ingestion CLI | `stockpred run-daily --since 2024-01-01` |
| Individual CLI steps | `stockpred ingest-prices --since 2024-01-01`, `stockpred compute-features`, `stockpred ingest-news`, `stockpred score-sentiment` |
| Start Postgres locally | `docker compose up -d postgres` |
| Apply migrations | `alembic upgrade head` |
| Generate a new migration | `alembic revision --autogenerate -m "describe change"` |
| Render migration SQL offline (no DB needed) | `alembic upgrade head --sql` |

Tests use an in-memory SQLite (see [tests/conftest.py](tests/conftest.py)); they do **not** require a running Postgres. Production code uses Postgres — the dialect difference is handled by `packages.shared.db.upsert_ignore`.

## Architecture

This is a **phase-based monorepo**. Only Phase 1 is implemented; Phases 2–4 are designed in `docs/` and exist only as written contracts.

**Implemented (Phase 1):** data ingestion → Postgres.
**Deferred (Phases 2–4):** XGBoost model + Backtrader backtest → signal engine + Alpaca paper broker → GCP deployment + monitoring.

Always read [docs/architecture.md](docs/architecture.md) before adding a new component — it states which layers exist, the contracts between them, and which directory each future service will live in. [docs/decisions/](docs/decisions/) holds ADRs explaining the *why* of major choices.

### Layout

- `services/<name>/` — deployable units. Today only `services/ingestion/` exists.
- `packages/shared/` — config (`pydantic-settings`), DB engine/session, ORM models, logging. **All services depend on this; nothing in `packages/shared/` should import from `services/`**.
- `migrations/` — single Alembic history for the whole DB schema. One migration per logical change; never edit a migration after it lands on `main`.
- `tests/` — flat layout, one test module per ingestion module.
- `docs/` — architecture, per-phase plans (each marked **implemented**/**DEFERRED**), and ADRs.

### Key conventions

1. **Idempotency is a contract.** Every ingest, feature compute, and sentiment scoring call must be safe to re-run with the same inputs and produce zero new rows on the second run. Tests assert this. When adding a new ingestion path, use `packages.shared.db.upsert_ignore()` rather than rolling your own insert.

2. **Long-form storage for indicators.** The `indicators` table is `(symbol, ts, name, value)` — one row per metric. Adding a new indicator is a code change in `services/ingestion/features.py` plus a row in `INDICATOR_COLUMNS`; it is **not** a schema migration. See [ADR 0004](docs/decisions/0004-postgres-long-form-indicators.md).

3. **Dialect-agnostic upserts.** `upsert_ignore()` detects Postgres vs SQLite at runtime so the same code path serves production and tests. Don't import `sqlalchemy.dialects.postgresql.insert` directly in service code.

4. **CLI is the only entrypoint.** `services/ingestion/cli.py` is a `typer` app installed as the `stockpred` console script. Cron will eventually call `stockpred run-daily`. There are no other entry points — no `__main__.py` modules with their own argparse, no notebooks committed to the repo.

5. **pandas-ta column naming has churned across versions.** The `INDICATOR_COLUMNS` map in `services/ingestion/features.py` deliberately accepts both old (`BBL_20_2.0`) and new (`BBL_20_2.0_2.0`) Bollinger column names. If a future pandas-ta release renames again, extend the map rather than pinning a version. See [ADR 0002](docs/decisions/0002-pandas-ta-over-talib.md).

6. **VADER lexicon is generic.** Sentiment tests use vocabulary VADER actually knows (e.g. "great wonderful amazing", not "soars" or "beat estimates"). When writing sentiment tests, check `polarity_scores()` produces non-zero compound before asserting on direction. See [ADR 0003](docs/decisions/0003-vader-for-phase-1-sentiment.md).

7. **Pylint disable list lives in `pyproject.toml`.** Many disables are workarounds for pydantic + SQLAlchemy + module-level cached singletons confusing pylint's static analysis. Before adding a new `# pylint: disable=` inline, check whether the rule is already globally disabled.

### Database schema

5 tables, defined in [packages/shared/models.py](packages/shared/models.py) and migrated in [migrations/versions/0001_initial.py](migrations/versions/0001_initial.py):

- `tickers` — watchlist master
- `price_bars` — daily OHLCV, unique on `(symbol, ts)`
- `indicators` — long-form, unique on `(symbol, ts, name)`
- `news_items` — headlines, unique on `url`
- `sentiments` — one row per news item, tagged with `model` (currently `vader-1.0`)

The `model` column on `sentiments` exists so future scorers (FinBERT, LLMs) can coexist with VADER results without overwriting.

## CI

Two workflows in `.github/workflows/`:
- `pylint.yml` — runs `pylint` on every push, Python 3.12
- `test.yml` — runs `pytest` on push and PR, Python 3.12

Python 3.12+ is required because the only available `pandas-ta` releases on PyPI require it.

Both install `pip install -e ".[dev]"` first; do not assume pip-installable packages without updating `pyproject.toml`.
