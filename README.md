# Stock-Prediction

Algorithmic stock-trading system, built in phases. Phases 1–3.5 are implemented: data ingestion, ML model + backtest, signal engine, Alpaca paper broker, and a Next.js dashboard. Phase 4 (deployment + ops) is the next milestone.

For the full roadmap and architecture, see [docs/README.md](docs/README.md).

## Quickstart

```bash
# 1. Configure
cp .env.example .env
# fill in NEWSAPI_KEY, ALPACA_KEY_ID/SECRET, adjust WATCHLIST
# Need Alpaca paper-trading keys? See:
# docs/phase-3-execution.md#getting-alpaca-paper-trading-keys

# 2. Start Postgres
docker compose up -d postgres

# 3. Install Python deps
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 4. Install Node deps (for the dashboard)
npm install

# 5. Apply migrations
alembic upgrade head

# 6. Run the pipeline end-to-end
stockpred run-daily --since 2024-01-01     # ingest data
stockpred train --model-version v1          # train model
stockpred run-signals --model-version v1    # generate signals
stockpred reconcile                          # sync with Alpaca

# 7. Browse the dashboard
npm run dev                                  # FastAPI + Next.js together
# → http://localhost:3000
```

Or invoke the Python pipeline step-by-step:

```bash
stockpred ingest-prices --since 2024-01-01
stockpred compute-features
stockpred ingest-news
stockpred score-sentiment
```

## Layout

- [services/ingestion/](services/ingestion/) — Phase 1 pipeline (prices, features, news, sentiment, CLI)
- [services/model/](services/model/) — Phase 2 model training, prediction, pandas backtester
- [services/signal/](services/signal/) — Phase 3 signal engine
- [services/broker/](services/broker/) — Phase 3 Alpaca client + reconciler
- [services/api/](services/api/) — Phase 3.5 FastAPI shim for the dashboard
- [apps/dashboard/](apps/dashboard/) — Phase 3.5 Next.js 15 dashboard
- [packages/shared/](packages/shared/) — config, DB, ORM models, logging used across services
- [migrations/](migrations/) — Alembic schema migrations
- [tests/](tests/) — pytest suite (SQLite in-memory; no external services required)
- [docs/](docs/) — architecture, phase plans, ADRs

## CI

[.github/workflows/ci.yml](.github/workflows/ci.yml) runs pylint and pytest in parallel jobs on push and PR.

## License

MIT
