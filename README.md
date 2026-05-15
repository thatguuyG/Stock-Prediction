# Stock-Prediction

Algorithmic stock-trading system, built in phases. **Phase 1 (current)** delivers the data foundation: daily OHLCV ingestion, technical indicators, news headlines, and per-headline sentiment — all persisted to Postgres. Phases 2–4 layer on a model, a signal engine, broker execution, and live deployment.

For the full roadmap and architecture, see [docs/README.md](docs/README.md).

## Quickstart

```bash
# 1. Configure
cp .env.example .env
# fill in NEWSAPI_KEY (https://newsapi.org — free tier) and adjust WATCHLIST

# 2. Start Postgres
docker compose up -d postgres

# 3. Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 4. Apply migrations
alembic upgrade head

# 5. Run the pipeline
stockpred run-daily --since 2024-01-01
```

Individual steps:

```bash
stockpred ingest-prices --since 2024-01-01
stockpred compute-features
stockpred ingest-news
stockpred score-sentiment
```

## Layout

- [services/ingestion/](services/ingestion/) — Phase 1 pipeline (prices, features, news, sentiment, CLI)
- [packages/shared/](packages/shared/) — config, DB, ORM models, logging used across services
- [migrations/](migrations/) — Alembic schema migrations
- [tests/](tests/) — pytest suite (SQLite in-memory; no external services required)
- [docs/](docs/) — architecture, phase plans, ADRs

## CI

- [.github/workflows/pylint.yml](.github/workflows/pylint.yml) — lints all Python on push
- [.github/workflows/test.yml](.github/workflows/test.yml) — runs pytest on push and PR

## License

MIT
