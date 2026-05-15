# ADR 0004 — Long-form indicators table

**Status:** Accepted (2026-05-15)

## Context

We need to persist technical indicators (RSI, MACD components, Bollinger Bands, SMAs, EMAs — twelve series in Phase 1). Two natural shapes:

**Wide table** — one column per indicator:

```sql
indicators_wide (symbol, ts, rsi_14, macd, macd_signal, macd_hist, bb_lower, ...)
```

**Long table** — one row per (symbol, ts, indicator name):

```sql
indicators (symbol, ts, name, value)
UNIQUE(symbol, ts, name)
```

## Decision

Use the **long form**.

## Consequences

**Positive**
- Adding a new indicator is a code change, not a schema change. No ALTER TABLE, no migration, no downtime risk on a large table.
- Per-indicator queries (`WHERE name = 'rsi_14'`) hit a covering index cleanly.
- Sparse coverage is natural: an SMA(200) doesn't exist for the first 199 bars; in wide form those are NULLs, in long form they simply aren't rows.
- Backfilling a newly-added indicator only writes the missing rows; existing data is untouched.

**Negative**
- Building the feature matrix requires pivoting at read time. The Phase 2 model code will do this with `pandas.pivot_table` or a SQL `crosstab`. Acceptable while the data fits comfortably in memory; will revisit if we move to billions of rows.
- Row count is ~12× the wide form. Doesn't matter at Phase 1 scale (a few hundred thousand rows total), and Postgres handles it without strain.
- Type is uniform `float` — can't mix indicator types in the same table (e.g. boolean regime flags). If we need that, we'll add a separate table for it rather than denormalising into a JSON column.

## Alternatives considered

- **Wide table**: faster reads for the "give me all features for date X" query, but every new indicator becomes a migration. Rejected — premature optimisation for a system that doesn't yet have a model.
- **JSONB column** (`indicators (symbol, ts, values JSONB)`): flexible but loses per-indicator indexing. Rejected.

## Revisit if

- A future model needs sub-second access to wide feature rows and the pivot becomes the bottleneck. Materialise a wide view from the long table at that point rather than changing the source-of-truth schema.
- We add non-numeric indicator outputs (e.g. categorical regime labels) — likely add a sibling `indicator_labels` table rather than widening this one.
