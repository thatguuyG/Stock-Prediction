# Architecture

## Target data flow

```
  ┌─────────────────────────────────────────────────────────────┐
  │ External sources                                            │
  │   yfinance (OHLCV)   NewsAPI (headlines)   Reddit/X (later) │
  └────────────┬──────────────────┬─────────────────────────────┘
               │                  │
               ▼                  ▼
  ┌─────────────────────────────────────────┐
  │ Ingestion service (Python)              │  ✅ Phase 1
  │   prices.py · features.py · news.py     │
  │   sentiment.py · cli.py (typer)         │
  └────────────────────┬────────────────────┘
                       │ SQL upserts (idempotent)
                       ▼
  ┌─────────────────────────────────────────┐
  │ Postgres                                │  ✅ Phase 1
  │   tickers · price_bars · indicators     │
  │   news_items · sentiments               │
  └────────────────────┬────────────────────┘
                       │ feature matrix queries
                       ▼
  ┌─────────────────────────────────────────┐
  │ Model service (XGBoost, later LSTM)     │  ✅ Phase 2
  │   training, walk-forward CV, inference  │
  └────────────────────┬────────────────────┘
                       │ probability scores
                       ▼
  ┌─────────────────────────────────────────┐
  │ Signal engine                           │  ✅ Phase 3
  │   model · indicator · sentiment · risk  │
  │   → buy / sell / hold + size            │
  └────────────────────┬────────────────────┘
                       │ orders
                       ▼
  ┌─────────────────────────────────────────┐
  │ Broker adapter — Alpaca paper           │  ✅ Phase 3
  │   submit · poll · reconcile             │
  └────────────────────┬────────────────────┘
                       │ fills, positions, P&L
                       ▼
  ┌─────────────────────────────────────────┐
  │ Dashboard (Next.js / read-only)         │  ✅ Phase 3.5
  │   positions · signals · equity curve    │
  │   ↕ FastAPI shim (services/api)         │
  └─────────────────────────────────────────┘
```

## Layer status

| Layer | Status | Notes |
|---|---|---|
| Data ingestion (prices, news) | ✅ implemented | `services/ingestion/{prices,news}.py` |
| Feature engineering (indicators) | ✅ implemented | `services/ingestion/features.py` (pandas-ta) |
| Sentiment scoring | ✅ implemented | `services/ingestion/sentiment.py` (VADER) |
| Storage | ✅ implemented | Postgres 16; 5-table schema; Alembic migrations |
| Model training / inference | ✅ implemented | XGBoost baseline; `services/model/{train,predict}.py` |
| Backtesting | ✅ implemented | Pure-pandas backtester ([ADR 0005](decisions/0005-pandas-backtester-over-backtrader.md)); `services/model/backtest.py` |
| Signal engine | ✅ implemented | `services/signal/rules.py` + `runner.py`; rationale logged to `signals.rationale` JSON |
| Broker execution | ✅ implemented | Thin Alpaca client (`services/broker/alpaca.py`); bracket orders; reconciliation loop |
| Dashboard | ✅ implemented | Next.js 15 App Router at `apps/dashboard/`; FastAPI shim at `services/api/` |
| Live deployment | ❌ deferred → Phase 4 | GCP Cloud Run + Cloud Scheduler |
| Monitoring / alerting | ❌ deferred → Phase 4 | TBD (likely GCP-native) |

## Component contracts

Each later component depends only on the contract of the layer below it. Capturing the contracts now lets each phase land independently.

### Ingestion → Storage (Phase 1, live)

- **Input:** `WATCHLIST` env var (comma-separated tickers), `--since` ISO date.
- **Output:** Idempotent upserts to `price_bars`, `indicators`, `news_items`, `sentiments`.
- **Guarantee:** Re-running with the same arguments produces zero new rows.

### Storage → Model (Phase 2 contract)

- **Input:** `SELECT … FROM price_bars JOIN indicators JOIN sentiments` for a date range.
- **Output:** Trained model artifact + per-(symbol, date) probability score written to a `predictions` table.
- **Guarantee:** No look-ahead bias — training only sees rows with `ts <= cutoff`.

### Model → Signal engine (Phase 3 contract)

- **Input:** Latest `predictions` row + latest indicators + current position state.
- **Output:** `BUY` / `SELL` / `HOLD` decision with target notional.
- **Guarantee:** Decisions are pure functions of inputs (replayable for audit).

### Signal engine → Broker (Phase 3 contract)

- **Input:** Signal + current Alpaca positions.
- **Output:** Alpaca order intents, then reconciliation rows in `trades`.
- **Guarantee:** Bracketed orders (stop-loss + take-profit) attached at submission.

## Repo layout (target)

```
.
├── docker-compose.yml
├── pyproject.toml
├── alembic.ini · migrations/
├── docs/                       — this folder
├── services/
│   ├── ingestion/              ✅ Phase 1
│   ├── model/                  ✅ Phase 2
│   ├── signal/                 ✅ Phase 3
│   ├── broker/                 ✅ Phase 3
│   └── api/                    ✅ Phase 3.5 (FastAPI shim)
├── apps/
│   └── dashboard/              ✅ Phase 3.5 (Next.js 15)
├── packages/shared/            — config, db, models, logging
└── tests/
```

## Design principles

- **Long-form > wide tables** for indicators and predictions — lets us add new features without DDL migrations. See [ADR 0004](decisions/0004-postgres-long-form-indicators.md).
- **Idempotent everything** — every ingest, every feature recompute, every prediction write is safe to re-run. Failed jobs do not corrupt state.
- **Local-first** — Postgres in Docker, SQLite in tests, no cloud account required to develop Phase 1.
- **Contracts, not abstractions** — each layer is a small Python module with a few functions. We add `Protocol` / interfaces only when a second implementation appears.
