# Phase 2 — Model & Backtest

> **Status: implemented.**
> Module home: [services/model/](../services/model/). Schema: predictions + backtest_runs (migration [0002_model_tables.py](../migrations/versions/0002_model_tables.py)).

## Goal

Turn the Phase 1 data lake into a working predictive model with honest, leak-free backtest numbers. The target acceptance criterion remains: walk-forward out-of-sample Sharpe materially above buy-and-hold over a 2-year window on real data.

## What's in

- Feature matrix built from `price_bars` + `indicators` + `sentiments`. Source: [services/model/features.py](../services/model/features.py).
- Walk-forward CV training with XGBoost — no random shuffle, no look-ahead. Source: [services/model/train.py](../services/model/train.py).
- Per-`(symbol, ts, model_version)` predictions persisted to the `predictions` table.
- Inference on demand via `stockpred predict`. Source: [services/model/predict.py](../services/model/predict.py).
- **Pure-pandas** backtester (Backtrader deferred; see [ADR 0005](decisions/0005-pandas-backtester-over-backtrader.md)). Source: [services/model/backtest.py](../services/model/backtest.py).
- Three new CLI subcommands mounted on the existing `stockpred` console script.

## CLI

```bash
stockpred train     --model-version v1 [--since 2020-01-01] [--train-window 252] [--val-window 63] [--step 63]
stockpred predict   --model-version v1 [--since 2020-01-01]
stockpred backtest  --model-version v1 [--threshold 0.55] [--slippage-bps 5] [--max-weight 0.20]
```

`train` does walk-forward CV, writes per-fold validation predictions, then retrains on the full dataset and saves the final model to `models/<version>.joblib`.
`predict` loads that artifact and scores any (symbol, ts) rows in the current feature matrix.
`backtest` reads `predictions` for a model_version, joins with `price_bars`, runs the long-only policy, and writes one row per run to `backtest_runs`.

## Locked-in defaults

| Decision | Value | Why |
|---|---|---|
| Walk-forward windows | train=252, validate=63, step=63 trading days | ~1y / 1q / 1q step |
| Long signal threshold | `score > 0.55` | small edge above 50% |
| Slippage | 5 bps per `|Δweight|` | matches Alpaca-era cost model |
| Commission | 0 | Alpaca paper is commission-free |
| Position sizing | Equal-weight across signals, capped 20%/name | concentration cap |
| Sentiment model | VADER | Phase 1 carry-over; FinBERT deferred |
| Topology | Single global model | per-symbol deferred |
| Model artifact | `joblib` in `models/<version>.joblib` (gitignored) | simple, replaceable later |

## Schema additions

```sql
predictions (
  id PK, symbol FK(tickers), ts, model_version, score FLOAT, label_pred INT
)
UNIQUE(symbol, ts, model_version), INDEX(symbol, ts), INDEX(model_version)

backtest_runs (
  id PK, model_version, started_at, finished_at,
  threshold FLOAT, sharpe FLOAT, max_drawdown FLOAT,
  hit_rate FLOAT, total_return FLOAT, turnover FLOAT,
  n_trades INT, notes TEXT
)
INDEX(model_version)
```

`predictions.model_version` lets variants coexist (e.g. `v1` and `v2_with_macro`) for A/B comparison.

## Feature engineering

For each `(symbol, ts)` row in the feature matrix:

| Family | Columns | Source |
|---|---|---|
| Indicators (wide) | `rsi_14`, `macd`, `macd_hist`, `macd_signal`, `bb_lower`, `bb_mid`, `bb_upper`, `sma_20`, `sma_50`, `sma_200`, `ema_12`, `ema_26` | Pivot of long-form `indicators` |
| Returns | `ret_1d`, `ret_5d`, `ret_20d` (log returns) | Computed on `price_bars` |
| Sentiment | `sent_mean_1d`, `sent_mean_3d`, `sent_mean_7d` (trailing mean compound) | Daily mean of `sentiments` joined to `news_items.published_at`; missing days = 0 |
| Volume | `vol_zscore_20d` (rolling 20-day z-score) | `price_bars.volume` |

Target: `1` if `close[t+1] > close[t]`, else `0`. Most-recent row per symbol is dropped at training time. Lags and rolling stats use `groupby(symbol)` so they never bleed across tickers.

## Walk-forward CV protocol

1. Pull the feature matrix; sort by `ts` ascending.
2. Slide a `(train_window, val_window)` pair forward by `step` trading days.
3. Each fold: train XGBoost on `[train_start, train_end)`, score on `[train_end, val_end]`.
4. Assert `max(train_ts) < min(val_ts)` per fold; raise if violated.
5. Concatenate all validation-window predictions → upsert into `predictions`.
6. Retrain on the full dataset; save the final `.joblib`.

XGBoost params (`services/model/train.py:DEFAULT_PARAMS`):

```python
n_estimators=200, max_depth=4, learning_rate=0.05,
subsample=0.8, colsample_bytree=0.8,
random_state=42, n_jobs=-1, eval_metric="logloss"
```

Hyperparameter tuning is deliberately deferred — the goal here is a defensible baseline, not a tuned model.

## Backtester

Vectorized in pandas, no event loop. Source: [services/model/backtest.py](../services/model/backtest.py).

1. Join `predictions` with `price_bars` on `(symbol, ts)`.
2. `signal[t, sym] = 1` if `score > threshold`, else `0`.
3. Daily equal-weight: `weight = min(1/N_long_today, max_weight)`. Long-only for now.
4. Daily portfolio return = `Σ_sym weight[t, sym] × (close[t+1]/close[t] - 1)`.
5. Subtract `slippage_bps × Σ |Δweight|` from each day's return.
6. Aggregate: Sharpe (annualised over 252), max drawdown, hit rate, total return (compounded), turnover (sum of `|Δweight|`), trade count.

## Metrics module

`services/model/metrics.py` — small, hand-rolled, **no sklearn dependency**:

- `sharpe(returns, periods_per_year=252)`
- `max_drawdown(equity_curve)`
- `hit_rate(returns)`
- `total_return(returns)` — compounded
- `log_loss(y_true, y_prob)` — binary cross-entropy
- `accuracy(y_true, y_pred)`

## Tests

| File | What it covers |
|---|---|
| [tests/test_model_features.py](../tests/test_model_features.py) | Feature matrix columns, warmup-row dropping, target correctness, sentiment-default-zero, per-symbol isolation, `--since` filter |
| [tests/test_model_metrics.py](../tests/test_model_metrics.py) | Sharpe / drawdown / hit rate / total return / log-loss against hand-computed values |
| [tests/test_model_train.py](../tests/test_model_train.py) | Synthetic embedded-signal dataset (next-day direction deterministic from `rsi_14 < 30`) → model accuracy ≥ 60%, predictions written, leakage assertion holds, re-train is idempotent |
| [tests/test_model_backtest.py](../tests/test_model_backtest.py) | Tiny deterministic panel, total return / hit rate / turnover match hand-computed values; `backtest_runs` row persisted; threshold filter works; missing predictions raises |

CI never trains on real market data — see [ADR 0006](decisions/0006-walk-forward-cv-windows.md) for the synthetic-data rationale.

## Real-world acceptance (not gated by CI)

After running `stockpred run-daily` over a few years of history, the real-data acceptance is:

- Walk-forward Sharpe > 0.5 on the watchlist over the most recent 2 years.
- Hit rate > 52%.
- Zero leakage warnings.

These are evaluated manually after you populate Phase 1 data.

## Predecessor / successor

- **Depends on:** [Phase 1](phase-1-data-and-signals.md) (data, indicators, sentiment populated in Postgres).
- **Unlocks:** [Phase 3](phase-3-execution.md) — the signal engine consumes the `predictions` table.

## Deliberately out of scope

- FinBERT sentiment (still VADER).
- FRED / macro features.
- Per-symbol models.
- HTML report generation.
- Hyperparameter tuning loop.
- Backtrader integration (covered by [ADR 0005](decisions/0005-pandas-backtester-over-backtrader.md)).
