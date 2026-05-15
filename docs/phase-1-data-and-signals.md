# Phase 1 — Data & Signals

**Status: implemented.** This doc is the operator's reference for the current build.

## Scope

- Daily OHLCV ingestion for a configurable watchlist via [yfinance](https://github.com/ranaroussi/yfinance).
- Technical indicator computation via [pandas-ta](https://github.com/twopirllc/pandas-ta).
- News headline ingestion via [NewsAPI](https://newsapi.org/).
- Per-headline sentiment scoring with [VADER](https://github.com/cjhutto/vaderSentiment).
- Idempotent persistence to Postgres.

## Database schema

5 tables, defined in [migrations/versions/0001_initial.py](../migrations/versions/0001_initial.py) and the ORM at [packages/shared/models.py](../packages/shared/models.py).

| Table | Purpose | Uniqueness |
|---|---|---|
| `tickers` | Master list of symbols we ingest | PK `symbol` |
| `price_bars` | Daily OHLCV bars | `(symbol, ts)` |
| `indicators` | Long-form technical indicators (one row per metric) | `(symbol, ts, name)` |
| `news_items` | Raw headlines | `url` |
| `sentiments` | VADER scores | `news_item_id` |

Long-form indicators were a deliberate choice — see [ADR 0004](decisions/0004-postgres-long-form-indicators.md).

## Indicators computed

All produced by `services.ingestion.features.compute_indicators()` and stored in the `indicators` table with the `name` column shown below.

| Metric | `indicators.name` | Source |
|---|---|---|
| RSI(14) | `rsi_14` | `pandas_ta.rsi(length=14)` |
| MACD | `macd`, `macd_signal`, `macd_hist` | `pandas_ta.macd(12, 26, 9)` |
| Bollinger Bands(20, 2) | `bb_lower`, `bb_mid`, `bb_upper` | `pandas_ta.bbands(20, 2)` |
| SMA | `sma_20`, `sma_50`, `sma_200` | `pandas_ta.sma` |
| EMA | `ema_12`, `ema_26` | `pandas_ta.ema` |

## Sentiment model

VADER 3.3.2 (lexicon + rules, no GPU). Each `news_item` gets one `sentiment` row tagged `model="vader-1.0"` with the four standard scores (`pos`, `neu`, `neg`, `compound`). See [ADR 0003](decisions/0003-vader-for-phase-1-sentiment.md) for why VADER instead of FinBERT.

## CLI

Entrypoint: `stockpred` (after `pip install -e .`). Source: [services/ingestion/cli.py](../services/ingestion/cli.py).

```bash
stockpred ingest-prices  --since 2024-01-01   # OHLCV for every WATCHLIST symbol
stockpred compute-features                    # indicators for all stored bars
stockpred ingest-news                         # NewsAPI headlines for every symbol
stockpred score-sentiment                     # VADER scores for any unscored news
stockpred run-daily      --since 2024-01-01   # all four in sequence (cron entry)
```

Every command is idempotent: re-running it with the same arguments produces zero new rows.

## Configuration

Loaded via `pydantic-settings` from `.env`. See [.env.example](../.env.example).

| Var | Purpose | Default |
|---|---|---|
| `DATABASE_URL` | Postgres connection string | `postgresql+psycopg://stockpred:stockpred@localhost:5432/stockpred` |
| `WATCHLIST` | Comma-separated US tickers | `AAPL,MSFT,NVDA` |
| `NEWSAPI_KEY` | https://newsapi.org/ free tier | *(empty — news ingest skipped if blank)* |
| `LOG_LEVEL` | Python logging level | `INFO` |

## Verification recipe

```bash
cp .env.example .env                    # then fill NEWSAPI_KEY
docker compose up -d postgres
pip install -e ".[dev]"
alembic upgrade head

stockpred ingest-prices --since 2024-01-01
psql "$DATABASE_URL" -c "SELECT count(*) FROM price_bars;"
# expected: ~340 × number of tickers

stockpred compute-features
psql "$DATABASE_URL" -c \
  "SELECT name, count(*) FROM indicators WHERE symbol='AAPL' GROUP BY name;"
# expected: 12 indicator names, hundreds of rows each

stockpred ingest-news && stockpred score-sentiment
psql "$DATABASE_URL" -c \
  "SELECT compound, headline FROM sentiments
   JOIN news_items ON news_items.id = sentiments.news_item_id LIMIT 5;"

pytest -q
```

Acceptance: a fresh clone reaches the end of this list with nothing but a `NEWSAPI_KEY` to fill in.

## Operational notes

- **NewsAPI free tier**: 100 requests/day, headlines limited to the last month. The client logs a warning on `429` and returns an empty list — does not crash the pipeline.
- **Failure mode**: yfinance occasionally returns empty DataFrames on transient API issues. The ingester logs a warning and skips that symbol; re-run to recover.
- **Backfill**: `--since` controls the earliest date. Re-running with an earlier date back-fills missing bars without touching existing ones.

## What's next (Phase 2 preview)

Build the feature matrix join that turns these three tables into a model-ready DataFrame, train an XGBoost classifier for next-day direction, and wire Backtrader for walk-forward evaluation. See [phase-2-model-and-backtest.md](phase-2-model-and-backtest.md).
