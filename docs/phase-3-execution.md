# Phase 3 — Execution

> **Status: DEFERRED — not yet implemented.**
> This document captures the intended design. Treat code references as forward-looking; nothing here exists yet.

## Goal

Turn model predictions into paper-traded orders against Alpaca, with explicit risk controls and a read-only dashboard for human oversight.

## Scope

- Signal engine: combine model score, technical confirmation, sentiment gate, and risk thresholds.
- Alpaca paper broker adapter: submit, poll, reconcile.
- Position sizing + stop-loss / take-profit.
- Read-only Next.js dashboard for positions, signals, recent fills.

## New components

```
services/signal/
├── rules.py            # decision logic
└── runner.py           # cron entrypoint: pull latest predictions, emit orders

services/broker/
├── alpaca.py           # thin adapter around alpaca-py
├── reconcile.py        # match local trade rows to broker fills
└── runner.py           # cron entrypoint: sync positions

services/dashboard/     # Next.js app
├── pages/
│   ├── index.tsx       # positions + P&L
│   ├── signals.tsx     # recent buys/sells with rationale
│   └── news.tsx        # latest headlines + sentiment
```

New tables:

```sql
orders   (id, symbol, side, qty, type, status, submitted_at, broker_order_id)
trades   (id, order_id, ts, qty, price, fee)
positions (symbol PK, qty, avg_price, updated_at)
risk_state (id, ts, equity, max_drawdown, exposure_pct)
```

## Signal rules (v1 sketch)

For each symbol with a fresh prediction:

1. **Filter**: skip if average daily volume < threshold or spread > threshold.
2. **Confirmation**: require model `score > 0.55` AND `close > sma_50` (long-only filter).
3. **Sentiment gate**: 7-day mean compound > -0.1 (don't fight obviously bad news).
4. **Sizing**: target 5% of equity per name; max 20 concurrent positions.
5. **Risk**: bracket order with 2% stop-loss and 4% take-profit; reject if total exposure would exceed 80%.

All rules are pure functions in `services/signal/rules.py` and exercised by deterministic unit tests.

## Broker integration

[alpaca-py](https://github.com/alpacahq/alpaca-py) for the paper trading API. Two cron jobs:

- **signal-tick** (every market-open weekday, after `compute-features` completes): generate signals, emit bracketed orders.
- **broker-tick** (every 5 min during market hours): poll Alpaca for fills, reconcile into `trades`/`positions`.

The two never share state in-process — they communicate only through the database. This makes either side individually restartable.

## Dashboard

Next.js read-only first. Talks to a FastAPI shim (`services/api/`, also Phase 3) that exposes:

- `GET /positions` — current portfolio.
- `GET /signals?limit=50` — recent decisions with the rule trace that justified them.
- `GET /equity-curve?from=...&to=...` — for charting.

No write endpoints in Phase 3 — order submission stays inside cron jobs. Manual overrides are a Phase 4 question.

## Acceptance criteria

- After 1 week of paper trading: orders flowing, no duplicate submissions on cron restart, positions reconcile within 5 min of fill.
- Dashboard renders current positions and last 50 signals in under 1s on a fresh load.
- Kill switch: `RISK_HALT=1` env var stops new order submission without crashing the cron job.

## Predecessor / successor

- **Depends on:** [Phase 2](phase-2-model-and-backtest.md) (`predictions` table populated).
- **Unlocks:** [Phase 4](phase-4-live-and-ops.md) (deploy this stack off the dev laptop).
