# Security Policy

## Scope

Stock-Prediction is a single-operator, self-hosted trading system. It holds broker
credentials and can submit orders, so the security surface is small but real:

- **Broker credentials** — `ALPACA_KEY_ID` / `ALPACA_SECRET_KEY` in `.env`.
- **The `stockpred` CLI** — the only path that submits orders (`run-signals`).
- **The FastAPI shim** (`services/api/`) — unauthenticated, read-only, bound to
  `127.0.0.1` by default.
- **The Next.js dashboard** (`apps/dashboard/`) — read-only viewer, no writes.

## Supported versions

Only `main` is supported. The project is versioned by phase, not by release tag;
there are no maintained backports.

| Version | Supported |
| ------- | --------- |
| `main`  | ✅ |
| Any older commit | ❌ |

## Operator responsibilities

These are the things most likely to hurt you, and none of them are enforced by code:

1. **Never commit `.env`.** It is gitignored and has never been committed — keep it
   that way. `.env.example` is the only file that belongs in git.
2. **Do not expose the API or dashboard publicly.** There is **no authentication and
   no authorization anywhere in this repo** — no accounts, no login, no API keys, no
   per-user data separation. Anyone who can reach `API_HOST:API_PORT` or the Next
   server can read the entire portfolio, every signal, and every order. The defaults
   (`127.0.0.1:8000`, CORS limited to `http://localhost:3000`) assume a single machine.
   Deploying beyond localhost requires adding auth first — see
   [docs/phase-4-live-and-ops.md](docs/phase-4-live-and-ops.md).
3. **Stay on paper trading.** `ALPACA_BASE_URL` should remain
   `https://paper-api.alpaca.markets`. Pointing it at the live API means real money
   against a model whose live behaviour differs from its backtest — see
   [docs/operations.md](docs/operations.md#4-before-you-trust-it-with-anything).
4. **Know the kill switch.** `RISK_HALT=1` in `.env` stops `run-signals` from
   submitting anything on its next invocation. It does not cancel orders already
   working at the broker — do that in the Alpaca dashboard.
5. **Rotate leaked keys at the broker.** Revoking an Alpaca key pair at
   [app.alpaca.markets](https://app.alpaca.markets/paper/dashboard/overview) is the
   only action that actually invalidates it; removing it from `.env` is not enough.

## Reporting a vulnerability

Report privately through GitHub — **Security → Advisories → Report a vulnerability**
on this repository. Please do not open a public issue for anything that could expose
credentials or allow unauthorized order submission.

Include what you'd need yourself: affected file or endpoint, the version or commit,
reproduction steps, and the impact you believe it has.

Expect an acknowledgement within **7 days** and a status update within **30 days**.
This is a personal project with no on-call rotation; if a fix is warranted it lands on
`main` and the advisory is published once it is available. If a report is declined,
you'll get the reasoning rather than silence.

## Out of scope

- Losses from trading decisions, model quality, or market behaviour. The system is
  provided as-is under the [MIT License](LICENSE) with no warranty; see the known
  gaps in [docs/operations.md](docs/operations.md#8-known-gaps).
- Vulnerabilities in Alpaca, NewsAPI, or yfinance — report those to their maintainers.
- The absence of authentication on the local API. That is a documented design
  decision for a localhost-only deployment ([ADR 0011](docs/decisions/0011-fastapi-thin-shim-over-orm-direct.md)),
  not a defect.
