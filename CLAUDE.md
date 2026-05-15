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
| Individual ingest steps | `stockpred ingest-prices --since 2024-01-01`, `stockpred compute-features`, `stockpred ingest-news`, `stockpred score-sentiment` |
| Train a model | `stockpred train --model-version v1 [--since 2020-01-01]` |
| Batch inference | `stockpred predict --model-version v1` |
| Run a backtest | `stockpred backtest --model-version v1 [--threshold 0.55]` |
| Start Postgres locally | `docker compose up -d postgres` |
| Apply migrations | `alembic upgrade head` |
| Generate a new migration | `alembic revision --autogenerate -m "describe change"` |
| Render migration SQL offline (no DB needed) | `alembic upgrade head --sql` |

Tests use an in-memory SQLite (see [tests/conftest.py](tests/conftest.py)); they do **not** require a running Postgres. Production code uses Postgres — the dialect difference is handled by `packages.shared.db.upsert_ignore`.

## Architecture

This is a **phase-based monorepo**. Phases 1 and 2 are implemented; Phases 3–4 are designed in `docs/` and exist only as written contracts.

**Implemented:**
- Phase 1 — data ingestion (`services/ingestion/`): prices, indicators, news, sentiment → Postgres.
- Phase 2 — model (`services/model/`): XGBoost walk-forward CV → predictions table; pure-pandas backtester → `backtest_runs`.

**Deferred:** Phase 3 (signal engine + Alpaca paper broker), Phase 4 (GCP deployment + monitoring + retraining).

Always read [docs/architecture.md](docs/architecture.md) before adding a new component — it states which layers exist, the contracts between them, and which directory each future service will live in. [docs/decisions/](docs/decisions/) holds ADRs explaining the *why* of major choices.

### Layout

- `services/<name>/` — deployable units. Today: `services/ingestion/` (Phase 1) and `services/model/` (Phase 2).
- `packages/shared/` — config (`pydantic-settings`), DB engine/session, ORM models, logging. **All services depend on this; nothing in `packages/shared/` should import from `services/`**.
- `migrations/` — single Alembic history for the whole DB schema. One migration per logical change; never edit a migration after it lands on `main`.
- `tests/` — flat layout, one test module per ingestion module.
- `docs/` — architecture, per-phase plans (each marked **implemented**/**DEFERRED**), and ADRs.

### Key conventions

1. **Idempotency is a contract.** Every ingest, feature compute, and sentiment scoring call must be safe to re-run with the same inputs and produce zero new rows on the second run. Tests assert this. When adding a new ingestion path, use `packages.shared.db.upsert_ignore()` rather than rolling your own insert.

2. **Long-form storage for indicators.** The `indicators` table is `(symbol, ts, name, value)` — one row per metric. Adding a new indicator is a code change in `services/ingestion/features.py` plus a row in `INDICATOR_COLUMNS`; it is **not** a schema migration. See [ADR 0004](docs/decisions/0004-postgres-long-form-indicators.md).

3. **Dialect-agnostic upserts.** `upsert_ignore()` detects Postgres vs SQLite at runtime so the same code path serves production and tests. Don't import `sqlalchemy.dialects.postgresql.insert` directly in service code.

4. **CLI is the only entrypoint.** `services/ingestion/cli.py` is the `typer` app installed as the `stockpred` console script. It mounts model commands via `services/model/cli.py:register(app)` at import time, so both Phase 1 (`ingest-prices`, …) and Phase 2 (`train`, `predict`, `backtest`) commands live on the same top-level CLI. There are no other entry points — no `__main__.py` modules with their own argparse, no notebooks committed to the repo.

5. **pandas-ta column naming has churned across versions.** The `INDICATOR_COLUMNS` map in `services/ingestion/features.py` deliberately accepts both old (`BBL_20_2.0`) and new (`BBL_20_2.0_2.0`) Bollinger column names. If a future pandas-ta release renames again, extend the map rather than pinning a version. See [ADR 0002](docs/decisions/0002-pandas-ta-over-talib.md).

6. **VADER lexicon is generic.** Sentiment tests use vocabulary VADER actually knows (e.g. "great wonderful amazing", not "soars" or "beat estimates"). When writing sentiment tests, check `polarity_scores()` produces non-zero compound before asserting on direction. See [ADR 0003](docs/decisions/0003-vader-for-phase-1-sentiment.md).

7. **Pylint disable list lives in `pyproject.toml`.** Many disables are workarounds for pydantic + SQLAlchemy + module-level cached singletons confusing pylint's static analysis. Before adding a new `# pylint: disable=` inline, check whether the rule is already globally disabled.

### Database schema

7 tables, defined in [packages/shared/models.py](packages/shared/models.py):

Phase 1 (migration `0001_initial.py`):
- `tickers` — watchlist master
- `price_bars` — daily OHLCV, unique on `(symbol, ts)`
- `indicators` — long-form, unique on `(symbol, ts, name)`
- `news_items` — headlines, unique on `url`
- `sentiments` — one row per news item, tagged with `model` (currently `vader-1.0`)

Phase 2 (migration `0002_model_tables.py`):
- `predictions` — per `(symbol, ts, model_version)`, with `score` and `label_pred`
- `backtest_runs` — one row per backtest invocation, with Sharpe / drawdown / hit rate / total return / turnover

The `model` column on `sentiments` and the `model_version` column on `predictions` exist so multiple model variants (VADER vs FinBERT, v1 vs v2) coexist without overwriting.

### Phase 2 conventions

8. **Idempotency extends to predictions.** `upsert_ignore()` is keyed on `(symbol, ts, model_version)`. Re-training the same `model_version` does **not** overwrite existing predictions — bump `--model-version` to record a new variant.

9. **Model artifacts live in `models/<version>.joblib`** and are gitignored. The trained XGBoost classifier is serialised together with `feature_columns` so the prediction code can re-validate inputs.

10. **Walk-forward CV defaults to 252/63/63 trading days.** Smaller test datasets must pass explicit `train_window`/`val_window`/`step` overrides — see [tests/test_model_train.py](tests/test_model_train.py).

11. **Synthetic-data testing for ML code.** Training/inference/backtest tests build a deterministic embedded-signal dataset directly in the SQLite test session — never call `yfinance` from a test. See [ADR 0006](docs/decisions/0006-walk-forward-cv-windows.md).

## CI

One workflow at `.github/workflows/ci.yml` with two parallel jobs:
- `lint` — `pylint` on all tracked Python (excluding `migrations/`)
- `test` — `pytest -q`

Both run on push and PR, both on Python 3.12. Python 3.12+ is required because the only available `pandas-ta` releases on PyPI require it.

Both install `pip install -e ".[dev]"` first; do not assume pip-installable packages without updating `pyproject.toml`.
