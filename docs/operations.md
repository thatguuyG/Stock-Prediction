# Operations Runbook — Running the System Day to Day

> **Status: living document.** This is the operator's guide: how to run the pipeline on a schedule, how a BUY on the dashboard turns into an actual trade, and how to read the numbers without fooling yourself.
>
> For *why* each component is shaped the way it is, see [architecture.md](architecture.md) and the [ADRs](decisions/).

---

## 1. The one thing to understand first

**The model predicts exactly one trading day ahead.**

The training target is `close[t+1] > close[t]` ([services/model/features.py](../services/model/features.py)), and the backtester measures the same one-day return. So a score of `0.76` does **not** mean "this is a good stock to own." It means:

> *76% estimated probability that this symbol closes higher on the **next trading day** after the bar the signal was computed from.*

Everything below follows from that. A signal has a **one-trading-day shelf life**. Acting on a signal dated more than one session back means trading on a forecast whose outcome is already public.

### Worked example

A BUY generated from Friday's bar is a prediction about **Monday's close**. If you see it on Tuesday, the answer is already known — Monday closed hours ago. Buying Tuesday captures none of the predicted move and enters at a price no gate ever evaluated.

This is why the schedule below is not optional garnish. It is the difference between acting on a forecast and acting on history.

---

## 2. Daily schedule

### Install

Two wrapper scripts handle the environment cron doesn't give you (PATH, virtualenv, working directory, logging):

| Script | Purpose |
|---|---|
| [scripts/daily.sh](../scripts/daily.sh) | Full pipeline: ingest → features → news → sentiment → predict → signals |
| [scripts/reconcile.sh](../scripts/reconcile.sh) | Pull fills, positions, and equity back from Alpaca |

```bash
chmod +x scripts/daily.sh scripts/reconcile.sh
crontab -e
```

```cron
CRON_TZ=America/New_York

# Nightly pipeline — 30 min after the 16:00 close, once the daily bar is final
30 16 * * 1-5  /path/to/Stock-Prediction/scripts/daily.sh

# Reconcile — after the open (catches overnight fills), midday, after close
45 9  * * 1-5  /path/to/Stock-Prediction/scripts/reconcile.sh
0  13 * * 1-5  /path/to/Stock-Prediction/scripts/reconcile.sh
10 16 * * 1-5  /path/to/Stock-Prediction/scripts/reconcile.sh
```

**Set `CRON_TZ=America/New_York`, not a UTC offset.** US market hours shift with American DST, which does not align with any other region's. Pinning the crontab to exchange time means the schedule tracks the market automatically instead of drifting an hour twice a year.

Edit the absolute `REPO` path at the top of both scripts if your checkout lives elsewhere.

### Why 16:30 ET

| When (ET) | What happens |
|---|---|
| 16:00 Mon | Market closes; Monday's daily bar finalizes |
| 16:30 Mon | `daily.sh` scores **Monday's bar** → prediction for Tuesday |
| 16:30 Mon | Any BUY submits as a market order, queued for the next open |
| 09:30 Tue | Order fills at Tuesday's open |
| 09:45 Tue | `reconcile` writes the fill → position appears on the dashboard |

Running before the close would score an unfinished bar. Running the next morning would produce a signal whose horizon has already elapsed.

### Verify it's working

```bash
tail -40 logs/daily-$(date +%F).log     # last night's run
stockpred report --limit 10             # positions, recent signals, risk state
```

A healthy run ends with `===== <timestamp> done =====` and a `run-signals:` summary line.

---

## 3. How a signal becomes a trade

### Automated (the default)

`run-signals` does the whole thing — you do not place the order yourself:

1. Reads the newest `predictions` row per symbol.
2. Applies the rule pipeline ([phase-3-execution.md](phase-3-execution.md#signal-rule-pipeline)) — score threshold, trend filter, sentiment gate, position/exposure caps.
3. Writes one `signals` row per symbol, **including HOLDs**, each with a JSON `rationale` recording every gate.
4. For BUY/SELL, sizes the position at 5% of equity and submits a **bracket order** to Alpaca: market entry, **2% stop-loss**, **4% take-profit**.
5. `reconcile` later mirrors fills into `positions` and `trades`.

Exits are automatic too. A position leaves the book when **any** of these fires:

- The stop is hit (−2%)
- The take-profit is hit (+4%)
- A later run scores it below 0.45 → SELL (`exit_on_bearish_score`)

### Manual (reading the dashboard yourself)

If you'd rather place trades by hand, run the pipeline in dry-run so it writes signals without touching the broker:

```bash
./scripts/daily.sh --dry-run
```

Then read `/signals` on the dashboard and act **at the next market open**, on rows dated the most recent trading session only. Ignore anything older — see §1.

To mirror what the automated path would have done: 5% of equity per name, stop at −2%, target at +4%.

### Decision reference

| Dashboard shows | Meaning | Action |
|---|---|---|
| `BUY` / `all_gates_passed` | Score > 0.55 and every gate passed | Enter at next open, 5% of equity |
| `SELL` / `exit_on_bearish_score` | Held, and score fell below 0.45 | Close the position at next open |
| `HOLD` / `score_below_threshold` | Model isn't confident enough | Do nothing |
| `HOLD` / `trend_filter_failed` | `close ≤ sma_50` — below trend | Do nothing |
| `HOLD` / `sentiment_gate_failed` | 7-day mean sentiment ≤ −0.1 | Do nothing |
| `HOLD` / `max_positions_reached` | Already at 20 open positions | Do nothing (capacity, not opinion) |
| `HOLD` / `exposure_cap_reached` | Adding this breaches 80% gross | Do nothing (capacity, not opinion) |
| `HOLD` / `hold_existing_position` | Position open, no exit trigger | Hold; don't add |

The two capacity reasons are worth distinguishing from the rest: `max_positions_reached` and `exposure_cap_reached` are **not** the model turning bearish. They mean the portfolio is full and this name lost the race for a slot. `SELECT rationale FROM signals WHERE symbol='X' AND ts='Y'` gives the full gate-by-gate trace.

### The kill switch

```bash
# in .env
RISK_HALT=1
```

`run-signals` then returns immediately without submitting anything. `reconcile` keeps running, so state tracking continues while you're halted, and each `risk_state` row records the halt flag.

---

## 4. Before you trust it with anything

The system will happily paper-trade a model nobody has validated. Do these first.

### Run a backtest

```bash
stockpred backtest --model-version v1
```

Nothing in the pipeline requires this, and no run happens automatically — so it is easy to be live for weeks against a model whose out-of-sample performance is unknown. The Phase 2 acceptance bar is Sharpe > 0.5 and hit rate > 52% ([phase-2-model-and-backtest.md](phase-2-model-and-backtest.md#real-world-acceptance-not-gated-by-ci)).

### Expect live results below backtest

The backtest assumes you capture `close[t] → close[t+1]`. Live, you fill at the **next open** — so the overnight gap, often a large share of the predicted move, is gone before you're in. This is a structural difference between the measurement and the execution, not a bug, and it biases live results downward. Do not read the gap between backtest and reality as a sign something is broken.

### Paper-trade first

Keep `--dry-run` on the daily cron for a week and compare each morning's decisions against what actually happened. `ALPACA_BASE_URL` should stay on `https://paper-api.alpaca.markets` — see [phase-3-execution.md](phase-3-execution.md#getting-alpaca-paper-trading-keys).

---

## 5. Data requirements

Walk-forward CV needs `train_window + val_window` = **315 distinct trading days in the feature matrix**, and the matrix is shorter than your raw price history because `sma_200` costs 200 bars of warm-up.

```
usable feature days ≈ price bars − 200
```

| History ingested | Feature days | Folds |
|---|---|---|
| ~400 bars (~1.5yr) | 200 | 0 — `train` fails |
| ~515 bars (~2yr) | 315 | 1 |
| ~1000 bars (~4yr) | 800 | 8 |
| ~1650 bars (~6.5yr) | 1450 | 19 |

**Ingest at least 4 years.** `--since 2020-01-01` is a good default:

```bash
stockpred ingest-prices --since 2020-01-01
stockpred compute-features
```

If `train` reports `Not enough data for walk-forward CV`, this is the cause. You *can* shrink the windows (`--train-window 120 --val-window 30 --step 30`) to smoke-test the pipeline, but use a throwaway `--model-version` — 120 days of training data for 19 features is not a model you should trade.

---

## 6. Command order matters

```
ingest-prices → compute-features → ingest-news → score-sentiment → predict → run-signals → reconcile
```

**`train` alone is not enough to produce tradeable signals.** It writes only *validation-fold* predictions, which stop at the last completed walk-forward fold — typically months behind today. `predict` is the batch-inference pass that scores current bars.

Skipping `predict` after a retrain is the single easiest way to break this system, and it fails **silently**: `run_once` inner-joins the newest prediction against the newest feature row on `(symbol, ts)`. Mismatched dates produce an empty join, zero signals, and a warning in the log — no error, no exit code. An empty dashboard looks identical to a quiet market.

Retraining periodically (monthly is reasonable) is a separate step, and always pairs with a fresh `predict`:

```bash
stockpred train --model-version v2
stockpred predict --model-version v2
stockpred backtest --model-version v2     # compare against v1 before switching
```

Then update the `--model-version` in `scripts/daily.sh`. Predictions are keyed on `(symbol, ts, model_version)`, so versions coexist and re-training an existing version will **not** overwrite its predictions — always bump the label.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Dashboard renders empty everywhere | API unreachable from the Next server | `curl 127.0.0.1:8000/healthz`; the pages `.catch(() => [])` so a failed fetch is indistinguishable from no data |
| `run-signals` reports `signals=0` | No `(symbol, ts)` overlap between predictions and features | Run `stockpred predict` — see §6 |
| Signals exist but are dated days back | Cron didn't run, or `predict` was skipped | Check `logs/daily-*.log`; **there is no staleness guard** — see §8 |
| `Not enough data for walk-forward CV` | Under 315 usable feature days | Ingest more history — see §5 |
| `number of parameters must be between 0 and 65535` | Batch exceeded Postgres' bind-parameter cap | Fixed by chunking in `upsert_ignore`; if it reappears, lower `MAX_BIND_PARAMS` in [packages/shared/db.py](../packages/shared/db.py) |
| `positions` empty but an order exists | Order submitted, not yet filled | Market orders placed after close fill at the next open; run `reconcile` after 09:45 ET |
| Alpaca 401 | Paper keys used against the live URL, or stale secret | See [phase-3-execution.md](phase-3-execution.md#getting-alpaca-paper-trading-keys) |

Useful queries:

```sql
-- what did the model see most recently?
SELECT symbol, ts::date, ROUND(score::numeric,4) FROM predictions
  WHERE model_version='v1' ORDER BY ts DESC LIMIT 10;

-- why didn't we buy X?
SELECT rationale FROM signals WHERE symbol='X' ORDER BY ts DESC LIMIT 1;

-- is my data current?
SELECT MAX(ts) FROM price_bars;
SELECT MAX(ts) FROM predictions;
```

The newest `predictions` timestamp should equal the newest `price_bars` timestamp. If predictions lag by exactly one bar, inference is dropping the latest row — see §8.

---

## 8. Known gaps

Things that will bite you, documented rather than silently fixed.

- **No staleness guard.** `_latest_predictions` ([services/signal/runner.py](../services/signal/runner.py)) takes the newest score per symbol *at any age*. If the host is down for a week, the next run trades a week-old forecast without complaint. Cron does not catch up on missed runs ([ADR 0008](decisions/0008-cli-cron-over-long-lived-scheduler.md)).
- **Holding period doesn't match the prediction horizon.** The model forecasts one day; the live position exits only on a bearish score or its ±4%/−2% bracket, so it can be held for weeks. The backtest measures daily rebalancing and therefore does **not** describe live behaviour.
- **Silent failure in the dashboard.** All three pages wrap their fetch in `.catch(() => [])`. Any API or DB outage renders as a normal empty dashboard.
- **Market holidays aren't modelled.** Cron fires Mon–Fri regardless. On a holiday no new bar arrives, so `predict` and `run-signals` re-score the previous bar — idempotent, but it does mean a stale-by-one-day signal can be re-emitted.
- **No advisory lock.** Two overlapping `run-signals` invocations are protected by upsert keys against duplicate *signals*, but not against double submission to Alpaca on the same `signal_id`.

---

## 9. Related docs

- [phase-3-execution.md](phase-3-execution.md) — signal rules, order lifecycle, Alpaca setup
- [phase-2-model-and-backtest.md](phase-2-model-and-backtest.md) — features, walk-forward CV, backtester
- [phase-3.5-dashboard.md](phase-3.5-dashboard.md) — API + dashboard internals
- [architecture.md](architecture.md) — system map
