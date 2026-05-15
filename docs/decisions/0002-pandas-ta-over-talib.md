# ADR 0002 — pandas-ta over TA-Lib

**Status:** Accepted (2026-05-15)

## Context

Phase 1 needs a technical-indicator library. The two obvious candidates are:

- [TA-Lib](https://ta-lib.org/) — the industry standard, C library with Python bindings.
- [pandas-ta](https://github.com/twopirllc/pandas-ta) — pure-Python alternative that integrates as a DataFrame accessor.

Both compute the same family of indicators (RSI, MACD, Bollinger, SMA/EMA). The difference is operational.

## Decision

Use **pandas-ta**.

## Consequences

**Positive**
- No native compilation step. `pip install pandas-ta` works on every developer laptop and every CI runner without `apt install libta-lib0-dev`.
- DataFrame-native API (`df.ta.rsi(append=True)`) — code stays inside the pandas idiom.
- Installable on Cloud Run base images without extending the Dockerfile.

**Negative**
- Pure-Python is slower than TA-Lib's C kernels. Negligible at our scale (a few hundred symbols × a few thousand bars); we'd notice if we moved to intraday tick data on thousands of symbols.
- Column naming has churned across pandas-ta versions (e.g. `BBL_20_2.0` → `BBL_20_2.0_2.0`). We handle this with a name map in `services/ingestion/features.py` rather than pinning a single version.
- The library is in maintenance mode (less active than TA-Lib). If it bitrots against a future pandas, we may have to switch.

## Alternatives considered

- **TA-Lib**: faster but the compile-from-source story is brittle. Common cause of "works on my laptop, fails in CI." Rejected for Phase 1.
- **Hand-rolled indicators**: tempting but a category error — RSI and MACD have subtle off-by-one conventions and we'd rather inherit a library's choices than re-derive them.

## Revisit if

- pandas-ta stops working with a pandas major upgrade.
- We move to intraday data and Python-level indicator computation becomes a bottleneck.
- A specific exotic indicator we want is only available in TA-Lib.
