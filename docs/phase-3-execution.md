# Phase 3 — Execution

> **Status: implemented (signal engine + broker).** The Next.js dashboard piece is deferred to [Phase 3.5](phase-3.5-dashboard.md). What's live today is the headless paper-trading loop: prediction → rules → bracket order → Alpaca → reconcile.

## Goal

Turn the Phase 2 `predictions` table into actual paper orders against Alpaca, with explicit risk controls and an audit trail.

## What's in

- Signal engine: pure-function decision logic ([services/signal/rules.py](../services/signal/rules.py)) consuming model score + indicators + sentiment + portfolio state.
- Signal runner ([services/signal/runner.py](../services/signal/runner.py)): pulls latest predictions, applies rules, writes a row per decision to `signals` and an order per BUY/SELL to `orders`.
- Thin Alpaca paper-trading client ([services/broker/alpaca.py](../services/broker/alpaca.py)) — no `alpaca-py` dependency; see [ADR 0007](decisions/0007-thin-alpaca-client-over-alpaca-py.md).
- Bracket orders: 2% stop-loss / 4% take-profit attached at submission.
- Reconciler ([services/broker/reconcile.py](../services/broker/reconcile.py)): pulls broker state, mirrors `orders` / `positions` / writes `risk_state`.
- `RISK_HALT=1` kill switch: signal runner exits early without submitting orders.
- Terminal report ([services/signal/cli.py](../services/signal/cli.py) `report` command): prints positions, recent signals, latest risk state.

## CLI

```bash
stockpred run-signals  --model-version v1                   # cron entry; emits orders
stockpred run-signals  --model-version v1 --dry-run         # log decisions without hitting Alpaca
stockpred reconcile                                          # cron entry; syncs broker state
stockpred report       [--limit 50]                         # terminal summary
```

Recommended cron — use the wrapper scripts, not bare CLI calls, so cron's minimal
environment (no PATH, no virtualenv, no working directory) doesn't sink the run:

```cron
CRON_TZ=America/New_York

# Nightly pipeline — 30 min after the 16:00 close, once the daily bar is final
30 16 * * 1-5  /path/to/Stock-Prediction/scripts/daily.sh

# Reconcile — after the open, midday, after close
45 9  * * 1-5  /path/to/Stock-Prediction/scripts/reconcile.sh
0  13 * * 1-5  /path/to/Stock-Prediction/scripts/reconcile.sh
10 16 * * 1-5  /path/to/Stock-Prediction/scripts/reconcile.sh
```

Pin the crontab to `CRON_TZ=America/New_York` rather than a fixed UTC offset —
US market hours shift with American DST, so a hardcoded UTC schedule silently
drifts an hour twice a year.

**`run-signals` must be preceded by `stockpred predict`** in the same run.
`train` only writes walk-forward *validation* predictions, which stop at the last
completed fold; without a fresh inference pass the runner finds no
`(symbol, ts)` overlap with the current feature row and emits zero signals.
[scripts/daily.sh](../scripts/daily.sh) sequences this correctly.

Full operator guide: [operations.md](operations.md).

## Getting Alpaca paper-trading keys

The runner submits to Alpaca's **paper** endpoint by default (`ALPACA_BASE_URL=https://paper-api.alpaca.markets`). Paper trading is free, requires no funding, no KYC, and starts you with $100k of fake equity.

1. **Sign up / log in** at https://alpaca.markets.
2. **Switch to Paper Trading.** Top-right of the dashboard has a toggle between "Live Trading" and "Paper Trading" — make sure you're on **Paper**. Paper and live accounts have entirely separate keys, and a paper key submitted to the live URL (or vice versa) returns 401.
3. **Open the API keys panel.** On the Paper Trading dashboard there's a section titled **"Your API Keys"** (sometimes shown as a small key icon on the right sidebar). Click **"View"** or **"Generate New Key"**.
4. **Copy both values immediately into `.env`:**
   - `API Key ID` → `ALPACA_KEY_ID`
   - `Secret Key` → `ALPACA_SECRET_KEY`
   - The secret is shown **exactly once**. If you close the dialog without copying it, you have to regenerate (which invalidates the previous secret).
5. **Verify** by running:
   ```bash
   stockpred reconcile
   ```
   With valid keys you'll see `equity=... exposure=...%` printed. With bad keys the AlpacaClient raises an `AlpacaError` with a 401.

Notes:
- **Don't change `ALPACA_BASE_URL`** unless you're deliberately going live. Live keys are issued separately in the Live tab.
- You can reset your paper account's balance from the dashboard at any time.
- If `ALPACA_KEY_ID` / `ALPACA_SECRET_KEY` are empty, `stockpred run-signals` automatically degrades to `--dry-run` and writes signals without submitting orders, so a missing `.env` can never accidentally ship orders.

## Decisions locked in

| Decision | Value | Source |
|---|---|---|
| Scope this build | Signal + broker only | User-chosen; dashboard deferred |
| Dashboard | Terminal CLI report | User-chosen; defers JS toolchain |
| Alpaca client | Thin `requests`-based, ~150 lines we own | [ADR 0007](decisions/0007-thin-alpaca-client-over-alpaca-py.md) |
| Scheduler | Host cron via CLI subcommands | [ADR 0008](decisions/0008-cli-cron-over-long-lived-scheduler.md) |
| Signal audit trail | Every decision (incl. HOLD) → `signals` table with JSON `rationale` | [ADR 0009](decisions/0009-signal-rationale-as-jsonb.md) |
| Long-only | Yes; SELL only triggers when already long + bearish score | Per docs/phase-3 |
| Score threshold | `score > 0.55` for BUY, `score < 0.45` for SELL | Reuses Phase 2's `DEFAULT_THRESHOLD` |
| Confirmation | `close > sma_50` (trend filter) | Per docs/phase-3 |
| Sentiment gate | 7-day mean compound > -0.1 | Per docs/phase-3 |
| Position sizing | 5% target per name | Per docs/phase-3 |
| Concurrency cap | 20 open positions max | Per docs/phase-3 |
| Exposure cap | 80% gross | Per docs/phase-3 |
| Bracket | 2% stop-loss / 4% take-profit | Per docs/phase-3 |
| Risk halt | `RISK_HALT=1` env var; checked at top of `run-signals` | Per docs/phase-3 |

## Schema additions

Migration [0003_model_tables.py](../migrations/versions/0003_execution_tables.py) adds five tables.

```sql
signals (id, symbol, ts, model_version, score, decision, rationale JSON, created_at)
  UNIQUE(symbol, ts, model_version)
  INDEX(symbol, ts), INDEX(model_version), INDEX(decision)

orders (id, symbol, side, qty, order_type,
        limit_price, stop_price, take_profit,
        status, submitted_at, filled_at, broker_order_id UNIQUE, signal_id)
  INDEX(status), INDEX(symbol, submitted_at)

trades (id, order_id, ts, qty, price, fee, broker_trade_id UNIQUE)
  INDEX(order_id)

positions (symbol PK, qty, avg_price, updated_at, source)

risk_state (id, ts, equity, cash, exposure_pct,
            max_drawdown, n_open_positions, halted, notes)
  INDEX(ts DESC)
```

`signals.rationale` records every gate's input value and pass/fail flag plus a top-level `reason` string. Queryable on Postgres via `->>` operators; on SQLite via JSON1.

## Signal rule pipeline

For each `(symbol, ts)` candidate:

1. **Already long + bearish score** → `SELL` (exit).
2. **Already long + neutral/bullish** → `HOLD` (don't double up).
3. **Score ≤ threshold** → `HOLD`.
4. **`close ≤ sma_50`** → `HOLD` (trend filter).
5. **7-day sentiment ≤ floor** → `HOLD` (sentiment gate).
6. **Open positions ≥ max** → `HOLD` (concurrency cap).
7. **Exposure + target > cap** → `HOLD` (exposure cap).
8. Else → `BUY` at `target_pct`.

The rationale dict logged to `signals.rationale` records every gate's value and outcome — including the gates that weren't reached (so the audit trail tells you which decision short-circuited).

## Order submission

1. Convert `target_pct` × latest `risk_state.equity` → notional → integer qty at current close.
2. Insert `orders` row with `status="pending"`, `signal_id` link.
3. Submit bracket order to Alpaca (`order_class=bracket`, market entry, 2% stop, 4% take).
4. Store `broker_order_id` + Alpaca's returned status. If Alpaca rejects: `status="rejected"`, error logged.
5. If `alpaca=None` (CLI `--dry-run` or no API keys): mark `status="dry_run"` and skip submission. **No orders ever silently ship without keys.**

### Execution timing

Signals are generated after the close, so a market order submitted at 16:30 ET
is queued and **fills at the next session's open**, not at the close price the
rules were evaluated against. The model's target is close-to-close
(`close[t+1] > close[t]`), so the overnight gap is missed before entry. Expect
live results to trail the backtest for this reason alone — it is a property of
daily-bar execution, not a defect. See
[operations.md](operations.md#4-before-you-trust-it-with-anything).

## Reconciliation

Always-on background loop (cron every 5 min during market hours):

1. For each non-terminal order with a `broker_order_id`, GET `/v2/orders/{id}` and update `status` / `filled_at`.
2. For any partial or full fill, insert a `trades` row representing the *delta* since the previous reconcile.
3. GET `/v2/positions`, upsert into `positions` (source = `alpaca`). Anything we tracked that the broker no longer holds → zero out.
4. GET `/v2/account`, compute exposure, append a new `risk_state` row.

Reconciler is the **only** writer to `positions` and `risk_state`. Signal runner only reads them — keeps the two crons decoupled.

## Risk halt

Set `RISK_HALT=1` in environment. Next `run-signals` invocation:
- Logs a warning.
- Returns immediately with `halted=True`.
- Writes zero rows to `signals` or `orders`.

Reconciler still runs unchanged — `halted` is recorded on the resulting `risk_state` row for audit.

## Tests

| File | Coverage |
|---|---|
| [tests/test_signal_rules.py](../tests/test_signal_rules.py) | Every rule branch: low score → HOLD, all gates pass → BUY, exposure cap → HOLD, already long → HOLD/SELL, parametrised score boundaries |
| [tests/test_signal_runner.py](../tests/test_signal_runner.py) | Synthetic prediction + risk_state in SQLite session; `signals` and `orders` rows written; idempotent re-run; `RISK_HALT=1` skips processing; Alpaca submission records `broker_order_id`; existing long position holds |
| [tests/test_broker_alpaca.py](../tests/test_broker_alpaca.py) | `responses` mocks every endpoint; auth headers, bracket-order payload shape, retry-on-5xx then fail, immediate-fail-on-4xx |
| [tests/test_broker_reconcile.py](../tests/test_broker_reconcile.py) | MagicMock'd Alpaca; status updates, fill creates Trade, partial→full only inserts the delta, broker positions mirrored, broker drops a name → local zeroed, `risk_state` row written |
| [tests/test_report.py](../tests/test_report.py) | Typer CliRunner; report renders cleanly on empty and populated DB |

CI never touches the live Alpaca API.

## Real-world acceptance (manual)

Populate `.env` with your personal Alpaca paper keys, run a few `run-signals` + `reconcile` cycles. Acceptance:
- Orders appear in the Alpaca paper dashboard.
- Local `orders.status` matches what Alpaca reports.
- Local `positions` matches Alpaca's position list after reconcile.
- Toggling `RISK_HALT=1` immediately stops new submissions on the next `run-signals`.

## What's still out of scope (Phase 3.5+)

- Next.js dashboard + FastAPI shim. See [phase-3.5-dashboard.md](phase-3.5-dashboard.md).
- Short selling, pair trades.
- Intraday signals (we run daily).
- Manual override / kill-individual-position via API.
- Multi-account portfolio aggregation.

## Predecessor / successor

- **Depends on:** [Phase 2](phase-2-model-and-backtest.md) (`predictions` table populated).
- **Unlocks:** [Phase 4](phase-4-live-and-ops.md) — deploying this stack off the dev laptop.
