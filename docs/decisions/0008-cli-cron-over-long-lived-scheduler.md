# ADR 0008 — CLI + host cron over an in-process scheduler

**Status:** Accepted (2026-05-15)

## Context

Phase 3 introduces two recurring jobs:
- `run-signals` — once a day, after market close.
- `reconcile` — every 5 minutes during market hours.

We need a scheduling mechanism. The obvious options were:

1. CLI subcommands you wire into the host's crontab (or systemd timers, GCP Cloud Scheduler later).
2. A long-running process using `APScheduler` or `schedule` that orchestrates both internally.

## Decision

CLI subcommands. `stockpred run-signals` and `stockpred reconcile` exit when they're done; the host's scheduler decides when to invoke them.

## Consequences

**Positive**
- Matches the model Phase 1 already uses (`stockpred run-daily`). Operators learn one pattern.
- Easy to test: each CLI invocation is a fresh process with no scheduler state to clean up.
- Easy to deploy on different infrastructure — cron on a VPS today, Cloud Scheduler tomorrow, GitHub Actions schedule as a stopgap. The application code doesn't care.
- A failed run cannot leave the scheduler in a bad state — Python interpreter exit cleans up. The next cron tick is a fresh attempt.
- Logs go to wherever cron sends them (typically syslog) instead of competing with a custom Python logging setup.

**Negative**
- No in-process job catch-up: if the host is down at 21:30 UTC, that day's `run-signals` is skipped. Cron features like `anacron` can recover this if needed.
- Two crons coordinating only via the database means slightly more boilerplate (re-establish DB connection, re-read config, re-instantiate Alpaca client every invocation). At our cadence (12 invocations/hour) this cost is negligible.
- Distributed locking is the operator's problem — if you accidentally run two crons in parallel, the database upsert keys protect against duplicate signals/orders, but not against double-submission to Alpaca on the same `signal_id`. We'd need an advisory lock in `run-signals` if we ever shard.

## Alternatives considered

- **APScheduler in a long-running Python process**. Adds a dependency, adds runbook complexity ("is the scheduler process alive?"), and creates a single point of failure. Rejected — premature for a two-job system.
- **Celery beat**. Even more infra (broker, worker, beat process). Vast overkill. Rejected.
- **A single `stockpred loop` command that sleeps**. Tempting for simplicity, but a 24h sleeping process loses the "fresh interpreter per invocation" benefit and conflates scheduling concerns with business logic. Rejected.

## Revisit if

- We start needing sub-minute scheduling, where cron's 1-minute resolution stops being enough.
- The cost of repeatedly re-establishing DB / Alpaca connections becomes meaningful (e.g. if we start running every 30 seconds).
- We need real cross-host coordination (sharded execution across multiple workers), at which point a proper queue/scheduler beats cron.
