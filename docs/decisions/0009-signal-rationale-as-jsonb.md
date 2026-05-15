# ADR 0009 — `signals.rationale` as a JSON column

**Status:** Accepted (2026-05-15)

## Context

Every signal-engine decision (BUY, SELL, HOLD) gets persisted to the `signals` table so the audit trail can answer "why did we (not) trade X on day Y?" The decision logic has eight gates today (score, trend, sentiment, position cap, exposure cap, already-long, bearish exit) and we'll add more as the system evolves.

## Decision

Store the per-decision audit as a single `rationale` JSON column on `signals`. SQLAlchemy maps this to `JSONB` on Postgres and `JSON` (TEXT-backed) on SQLite via `sqlalchemy.types.JSON`.

The dict captures every gate's input value, its pass/fail flag, and a top-level `reason` string naming the gate that short-circuited the decision.

## Consequences

**Positive**
- Adding a new rule (Phase 3.x will add several) is a code change, no migration. New gates just produce new keys in `rationale`.
- Postgres `->>` and `@>` operators let us query rationales without joining auxiliary tables: `WHERE rationale->>'reason' = 'sentiment_gate_failed'` is enough to find every HOLD blamed on bad news.
- SQLite happily reads/writes the same data as JSON-encoded TEXT — tests work without database-specific code.
- The structure is self-documenting in a Postgres GUI or `psql`: anyone can `SELECT rationale FROM signals LIMIT 5` and understand what each decision saw.

**Negative**
- No schema enforcement on the contents of `rationale`. A typo in the runner could write `{"reasion": "..."}` and we'd only notice when nothing matches the query.
- Querying nested keys is slower than columnar storage. At our scale (one signal per (symbol, day, model_version) — thousands of rows total) it doesn't matter; if `signals` ever grows to millions we'd want indexed expression columns on the hottest keys.
- The `signals` table can no longer be straightforwardly dumped to CSV with one column per gate — anyone exporting for analysis needs to expand the JSON.

## Alternatives considered

- **Wide table with one column per gate.** Schema-enforced and indexable, but every new rule is an `ALTER TABLE`. Rejected for the same reason we picked long-form indicators in [ADR 0004](0004-postgres-long-form-indicators.md): we're optimising for ease-of-extension, not query latency.
- **Separate `signal_rationale_kv (signal_id, key, value)` long-form table.** Maximally flexible but slow to read (need to aggregate to reconstruct one decision) and clumsy to write (multiple inserts per signal). Rejected.
- **Free-text `reason` column only, no structured rationale.** Smallest change, but loses every gate's input value — you'd see "score_below_threshold" but couldn't query "what scores have we seen in HOLDs?" Rejected as a regression in observability.

## Revisit if

- The `rationale` column grows beyond ~1KB per row (today's payload is ~400 bytes). Bigger would mean we're recording too much.
- We need analytical queries on rationale fields fast enough that JSON parsing becomes the bottleneck — at that point we add a Postgres expression index on the hot keys.
- We start running a separate analytics pipeline (dbt? a warehouse?) and the wide-column form becomes natively easier there.
