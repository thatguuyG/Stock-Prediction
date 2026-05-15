# Phase 4 — Live & Ops

> **Status: DEFERRED — not yet implemented.**
> This document captures the intended design. Treat code references as forward-looking; nothing here exists yet.

## Goal

Move the system off a dev laptop onto managed infrastructure with monitoring, alerting, and a scheduled retraining cadence. Still paper-trading until the live-money decision is made deliberately.

## Scope

- Deploy to GCP Cloud Run (or equivalent) with Cloud Scheduler triggering cron jobs.
- Managed Postgres (Cloud SQL).
- Monitoring + alerting: Prometheus/Grafana, or GCP-native (Cloud Monitoring + Logging).
- Retraining pipeline: weekly walk-forward update of the production model.
- On-call runbook.

## Target topology

```
                   ┌──────────────────┐
                   │ Cloud Scheduler  │ — cron triggers
                   └────────┬─────────┘
                            │ HTTP
        ┌───────────────────┼────────────────────┐
        ▼                   ▼                    ▼
  ┌──────────┐        ┌──────────┐         ┌──────────┐
  │ ingest   │        │ signal   │         │ broker   │
  │  Cloud   │        │  Cloud   │         │  Cloud   │
  │   Run    │        │   Run    │         │   Run    │
  └────┬─────┘        └────┬─────┘         └────┬─────┘
       │                   │                    │
       └─────────────┬─────┴────────────────────┘
                     ▼
              ┌────────────┐
              │ Cloud SQL  │ ← managed Postgres 16
              └─────┬──────┘
                    │
                    ▼
             ┌────────────┐
             │ dashboard  │ ← Cloud Run + Vercel-style static for Next.js
             └────────────┘
```

## Deployments

Each service is a separate Cloud Run revision:

| Service | Image | Schedule | Min instances |
|---|---|---|---|
| `ingestion` | `services/ingestion/Dockerfile` | daily at market close + 30 min | 0 (cold-start ok) |
| `signal` | `services/signal/Dockerfile` | weekdays at market open | 0 |
| `broker` | `services/broker/Dockerfile` | every 5 min during market hours | 1 |
| `api` | `services/api/Dockerfile` | always on | 1 |
| `dashboard` | static export | always on | 1 |

CI/CD: GitHub Actions builds + deploys on `main` via `gcloud run deploy`. Each service has a smoke-test step that calls `/healthz` after deploy.

## Secrets

GCP Secret Manager, accessed via service account binding. No `.env` in containers.

- `DATABASE_URL`
- `NEWSAPI_KEY`
- `ALPACA_KEY_ID`, `ALPACA_SECRET_KEY`

## Monitoring

Per-service metrics emitted to Cloud Monitoring:

- Ingestion: rows inserted per run, last-success timestamp, error rate.
- Signal: signals emitted per run, decision distribution.
- Broker: orders submitted, fills received, position drift (local vs. Alpaca).
- Model: per-prediction latency, drift in score distribution vs. training set.

Dashboards: one per service plus a global "system health" board.

## Alerting

Page on (PagerDuty or just email for paper-only):

- Ingestion has not succeeded for > 48h.
- Broker reconciliation has unresolved discrepancy for > 30 min.
- Equity drawdown exceeds configured threshold.
- Any service is returning 5xx > 5% for 10 min.

## Retraining

Cloud Scheduler weekly job:

1. Pull latest `price_bars` + `indicators` + `sentiments`.
2. Re-run walk-forward training from [Phase 2](phase-2-model-and-backtest.md).
3. Compare new model's last-fold metrics against current production.
4. If improved by > X bps Sharpe → bump `model_version`; otherwise log and skip.
5. Always write the candidate model to GCS for offline analysis.

## On-call runbook (skeleton)

- **Ingestion stalled** → check NewsAPI quota; check yfinance status; re-run manually.
- **Reconciliation drift** → compare `positions` table vs. Alpaca portfolio; resync from Alpaca as source of truth.
- **Equity drawdown alert** → flip `RISK_HALT=1`, halt new orders, investigate before clearing.
- **Model drift** → check `score` histogram vs. training baseline in Grafana; force a retrain if obviously degraded.

## Acceptance criteria

- 30 days of unattended paper trading with no manual intervention required.
- All four cron jobs running on schedule with green health checks.
- One full retraining cycle completes successfully and produces a model artifact in GCS.

## Predecessor

- **Depends on:** [Phase 3](phase-3-execution.md) (working signal + broker stack to deploy).

## Out of scope here

- Going live with real capital. That is a separate decision with its own checklist (regulatory, tax, position limits, audit).
