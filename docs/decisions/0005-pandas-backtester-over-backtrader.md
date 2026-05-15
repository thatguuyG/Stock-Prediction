# ADR 0005 — Pure-pandas backtester over Backtrader

**Status:** Accepted (2026-05-15)

## Context

Phase 2 needs a backtester to evaluate the model walk-forward. The docs originally sketched [Backtrader](https://www.backtrader.com/) — a full event-driven engine with broker abstraction, strategy classes, and per-bar order simulation. That choice would have added meaningful surface area and a non-trivial dependency, and we hadn't yet asked whether we need any of it.

## Decision

Use a **pure-pandas vectorised backtester** in [services/model/backtest.py](../../services/model/backtest.py). No Backtrader dependency. ~150 lines of code.

## Consequences

**Positive**
- Zero new third-party packages.
- Vectorised pandas is fast — a 2-year, 10-symbol backtest runs in milliseconds.
- The whole P&L computation fits in one file you can read top-to-bottom.
- Tests against hand-computed expectations are tractable (we know exactly what `Σ weight × return` is).
- Slippage and turnover are first-class outputs without needing a broker simulator.

**Negative**
- No per-bar order simulation. We can't model partial fills, market-on-open vs market-on-close timing, or limit-order behaviour.
- No portfolio-aware features like rebalancing at month-end or strategy composition.
- The "strategy" is hard-coded in the function (long-only, equal-weight, capped). Adding short-selling, pair trades, or anything stateful means new code rather than a new strategy class.

## Alternatives considered

- **Backtrader** as originally specced. Rejected for Phase 2 — its value sits in Phase 3, when the signal engine starts caring about order types, fill timing, and broker behaviour. Pulling it in now buys us nothing the pandas implementation can't do.
- **vectorbt** — actively maintained, vectorised, and feature-rich. Rejected: adds 50MB+ of dependencies and an opinionated API we'd inherit before we know what we need.
- **Roll our own event loop**. Worst of both worlds.

## Revisit if

- Phase 3's signal engine needs to simulate market-on-open vs market-on-close orders, partial fills, or limit-order slippage models. At that point Backtrader earns its keep.
- We add short-selling, options, or anything non-linear in portfolio construction.
- We need to test execution timing against the actual Alpaca paper API and want to mirror its behaviour offline.
