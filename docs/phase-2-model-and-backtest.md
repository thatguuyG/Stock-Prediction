# Phase 2 — Model & Backtest

> **Status: DEFERRED — not yet implemented.**
> This document captures the intended design. Treat code references as forward-looking; nothing here exists yet.

## Goal

Turn the Phase 1 data lake into a working predictive model with honest, leak-free backtest numbers. Success at this phase = an XGBoost classifier whose out-of-sample Sharpe is materially above buy-and-hold on the same period.

## Scope

- Build a feature matrix from `price_bars`, `indicators`, and `sentiments`.
- Train an XGBoost classifier for next-day direction (up vs. down).
- Walk-forward cross-validation — no look-ahead, no random shuffling.
- Persist trained model artifacts and per-(symbol, date) predictions.
- Backtest via [Backtrader](https://www.backtrader.com/) with realistic costs and slippage.

## New components

```
services/model/
├── features.py        # feature matrix construction (joins indicators + sentiment lags)
├── train.py           # XGBoost training + walk-forward CV
├── predict.py         # batch inference to a `predictions` table
└── backtest.py        # Backtrader harness
```

New table:

```sql
predictions (id, symbol, ts, model_version, score, label_pred)
UNIQUE(symbol, ts, model_version)
```

`model_version` lets multiple model variants coexist for A/B comparison.

## Feature engineering

For each `(symbol, ts)` we build a row from:

- All indicators at `ts` (pivoted wide from the long-form `indicators` table).
- Lagged returns: 1d, 5d, 20d.
- Rolling sentiment aggregates: mean `compound` over last 1d / 3d / 7d windows from `sentiments` joined to `news_items.published_at`.
- Volume z-score over 20-day rolling window.

Target label: `1` if `close[t+1] > close[t]`, else `0`. We drop the most recent row at training time (no label).

## Walk-forward CV protocol

1. Sort all rows by date globally.
2. Train on `[t_0, t_1]`, validate on `[t_1, t_2]`.
3. Slide window forward; retrain.
4. Report aggregated metrics across folds:
   - Hit rate (accuracy)
   - Log-loss
   - Sharpe ratio of a long-only strategy that goes long when `score > 0.55`
   - Max drawdown
   - Turnover (proxy for transaction cost burden)

## Backtest contract

Backtrader strategy `model_score > threshold → long`, `< 1 - threshold → flat` (long-only for Phase 2). Includes:

- Per-trade commission of 0 (Alpaca commission-free) but a 5 bps slippage assumption.
- Position sizing: equal-weight across active signals, capped at 20% of equity per name.

Output: an HTML report (`reports/<run_id>.html`) plus a row in a `backtest_runs` table for tracking.

## Acceptance criteria

- Walk-forward Sharpe > 0.5 on the watchlist over the most recent 2 years.
- Hit rate > 52% (worse than random by enough to clear noise).
- No data leakage detected: a sanity test asserts that for each fold, no training timestamp ≥ the validation start.

## Open questions for when we start

- Do we still want VADER, or upgrade to FinBERT here? VADER was Phase 1's cheap default.
- Single global model vs. per-symbol models?
- Add macroeconomic features (FRED) now or later?

## Predecessor / successor

- **Depends on:** [Phase 1](phase-1-data-and-signals.md) (data + indicators + sentiment in Postgres).
- **Unlocks:** [Phase 3](phase-3-execution.md) (signal engine consumes the `predictions` table).
