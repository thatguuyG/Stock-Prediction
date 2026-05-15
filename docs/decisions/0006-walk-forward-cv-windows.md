# ADR 0006 — Walk-forward CV window sizes (252 / 63 / 63)

**Status:** Accepted (2026-05-15)

## Context

Phase 2's walk-forward cross-validation needs three numbers: training window, validation window, and step. They are knobs but they shape every metric the model reports, so the defaults matter — and they must be defensible.

## Decision

Default to **train_window = 252, val_window = 63, step = 63** trading days. Approximately 1 year train, 1 quarter test, step quarterly.

Caller can override per-fold via the `train` CLI flags or the function kwargs, but all CI numbers and the documented acceptance criterion use these defaults.

## Consequences

**Positive**
- One year of training data is enough for tree models to learn cross-sectional patterns without obvious staleness.
- Quarterly retraining matches how a human portfolio manager would refresh signals — small enough to catch regime shifts, big enough to avoid overfitting to last week's noise.
- 252 + 63 = 315 days minimum before fold 0 produces output. With 2+ years of data we get ~5 folds, enough to average over.
- Step = val_window means folds don't overlap on the validation side, so per-fold metrics are independent — averaging them is statistically reasonable.

**Negative**
- Tests need >= 315 days to even run one fold. Our synthetic-data test fixture explicitly works around this by passing smaller windows (`train_window=80, val_window=20, step=20`).
- Step = val_window throws away potential evaluation: with step < val_window we'd get more folds but with overlapping (correlated) validation periods.
- 252 is a US-trading-days approximation; international markets have ~250–253 trading days. We don't care for now.

## Alternatives considered

- **Expanding-window CV** (train_window grows with each fold) — gives the latest fold the most data, but training-set composition drifts in ways that confound fair comparison across folds. Rejected.
- **K-fold without time ordering** — guarantees data leakage in a financial time series. Rejected on first principles.
- **Walk-forward with step = 1 day** — many more folds, but folds become near-duplicates and the implicit assumption of independent metrics breaks. We can revisit if we add bootstrap-style confidence intervals later.

## Revisit if

- We move to intraday bars. The whole concept of "252 trading days" loses meaning — windows would be in hours.
- We adopt a regime-detection layer and want shorter training windows during volatile periods.
- We start trading thinly-covered names where 1 year of history isn't enough to learn anything stable.
