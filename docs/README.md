# Stock-Prediction Documentation

The system is built in phases. **Phase 1 is implemented today**; Phases 2–4 are designed but not yet built.

## Index

| Doc | Status | What's inside |
|---|---|---|
| [architecture.md](architecture.md) | living | Target system diagram, component contracts, which layers exist today |
| [phase-1-data-and-signals.md](phase-1-data-and-signals.md) | **implemented** | DB schema, CLI, indicators, sentiment model, verification recipe |
| [phase-2-model-and-backtest.md](phase-2-model-and-backtest.md) | **implemented** | XGBoost baseline, walk-forward CV, pure-pandas backtester, metrics |
| [phase-3-execution.md](phase-3-execution.md) | **implemented** (no dashboard) | Signal engine rules, Alpaca paper integration, order lifecycle, risk halt |
| [phase-3.5-dashboard.md](phase-3.5-dashboard.md) | **deferred** | FastAPI shim + Next.js dashboard (carve-out from Phase 3) |
| [phase-4-live-and-ops.md](phase-4-live-and-ops.md) | **deferred** | GCP Cloud Run deployment, scheduling, monitoring, retraining, alerts |

## Architecture Decision Records (ADRs)

Decisions taken so far. Each ADR follows: Context / Decision / Consequences / Alternatives.

- [decisions/0001-monorepo-layout.md](decisions/0001-monorepo-layout.md)
- [decisions/0002-pandas-ta-over-talib.md](decisions/0002-pandas-ta-over-talib.md)
- [decisions/0003-vader-for-phase-1-sentiment.md](decisions/0003-vader-for-phase-1-sentiment.md)
- [decisions/0004-postgres-long-form-indicators.md](decisions/0004-postgres-long-form-indicators.md)
- [decisions/0005-pandas-backtester-over-backtrader.md](decisions/0005-pandas-backtester-over-backtrader.md)
- [decisions/0006-walk-forward-cv-windows.md](decisions/0006-walk-forward-cv-windows.md)
- [decisions/0007-thin-alpaca-client-over-alpaca-py.md](decisions/0007-thin-alpaca-client-over-alpaca-py.md)
- [decisions/0008-cli-cron-over-long-lived-scheduler.md](decisions/0008-cli-cron-over-long-lived-scheduler.md)
- [decisions/0009-signal-rationale-as-jsonb.md](decisions/0009-signal-rationale-as-jsonb.md)

## How to read these docs

- The **architecture** doc is the map. Start there.
- The **phase-1** doc is the operator's reference for the current build.
- The **phase-2/3/4** docs are intent, not specification. They will evolve as each phase begins.
- ADRs capture *why* a choice was made so a future contributor can either defend or revisit it.
