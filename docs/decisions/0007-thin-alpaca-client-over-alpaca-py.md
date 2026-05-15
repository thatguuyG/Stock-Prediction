# ADR 0007 — Thin Alpaca HTTP client over `alpaca-py`

**Status:** Accepted (2026-05-15)

## Context

Phase 3 needs a paper-trading client. The official Python SDK [`alpaca-py`](https://github.com/alpacahq/alpaca-py) is the obvious default. We use exactly four endpoints: account, positions, submit order, get order.

## Decision

Roll our own thin client in [services/broker/alpaca.py](../../services/broker/alpaca.py) using `requests` (which we already depend on for the NewsAPI ingester). ~150 lines we own.

## Consequences

**Positive**
- No new dependency, no SDK version pin to worry about.
- `responses` library mocks every endpoint cleanly in tests — no need to learn alpaca-py's model layer.
- Retry-on-5xx with exponential backoff is implemented exactly how we want it.
- The client returns raw dicts; the reconciler picks the fields it needs without an intermediate model translation layer.

**Negative**
- We are now responsible for keeping in sync with Alpaca's API. If they add a new field we want, we need to add it.
- `alpaca-py`'s response models give type safety we forgo. We add a defensive `dict.get(...)` everywhere instead.
- Authentication is hard-coded to the two-header model — if Alpaca ever introduces OAuth or token-based auth, we'd need to rewrite.

## Alternatives considered

- **`alpaca-py`**: would buy us coverage of every Alpaca endpoint plus pydantic response models. Rejected for now — we use 4/100+ endpoints and the SDK pulls in transitive deps we don't need.
- **A `requests` retry adapter via `urllib3.Retry`**: cleaner than the hand-rolled retry loop, slightly more opaque. Reasonable swap-in if we ever want connection pooling configured. Not worth complexity now.

## Revisit if

- We add streaming (websocket) feeds — `alpaca-py` has battle-tested WebSocket clients that would take weeks to replicate.
- We expand beyond the four endpoints to a meaningful fraction of Alpaca's API surface (>15 endpoints).
- Alpaca makes a breaking change to their REST authentication.
