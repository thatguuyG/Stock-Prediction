"""Phase 3 broker CLI subcommands."""
from __future__ import annotations

import requests
import typer

from packages.shared.config import get_settings
from packages.shared.db import session_scope
from services.broker.alpaca import AlpacaClient, AlpacaError
from services.broker.reconcile import reconcile_once


def _mask(value: str) -> str:
    if not value:
        return "<empty>"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}…{value[-4:]} (len={len(value)})"


def register(app: typer.Typer) -> None:
    @app.command("reconcile")
    def reconcile_cmd() -> None:
        settings = get_settings()
        if not (settings.alpaca_key_id and settings.alpaca_secret_key):
            typer.echo(
                "ERROR: ALPACA_KEY_ID and ALPACA_SECRET_KEY must be set in .env to reconcile.",
                err=True,
            )
            raise typer.Exit(code=1)
        alpaca = AlpacaClient()
        with session_scope() as session:
            summary = reconcile_once(session, alpaca, halted=settings.risk_halt)
        typer.echo(
            f"reconcile: orders_updated={summary.n_orders_updated} "
            f"new_trades={summary.n_new_trades} positions={summary.n_positions} "
            f"equity={summary.equity:.2f} exposure={summary.exposure_pct:.2f}%"
        )
        if summary.errors:
            for err in summary.errors:
                typer.echo(f"  error: {err}", err=True)

    @app.command("check-alpaca")
    def check_alpaca_cmd() -> None:
        """Hit /v2/account directly and report what we see, masked. Useful for
        diagnosing 401s without re-running the full reconcile."""
        settings = get_settings()
        key = settings.alpaca_key_id
        secret = settings.alpaca_secret_key
        base = settings.alpaca_base_url.rstrip("/")

        typer.echo(f"base_url         : {base}")
        typer.echo(f"ALPACA_KEY_ID    : {_mask(key)}")
        typer.echo(f"ALPACA_SECRET_KEY: {_mask(secret)}")

        if key != key.strip() or secret != secret.strip():
            typer.echo(
                "WARN: key or secret has leading/trailing whitespace — strip it in .env",
                err=True,
            )

        if not (key and secret):
            typer.echo("ERROR: key or secret is empty.", err=True)
            raise typer.Exit(code=1)

        url = f"{base}/v2/account"
        headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        try:
            resp = requests.get(url, headers=headers, timeout=10)
        except requests.RequestException as exc:
            typer.echo(f"ERROR: network failure calling {url}: {exc}", err=True)
            raise typer.Exit(code=2) from exc

        typer.echo(f"GET /v2/account  : HTTP {resp.status_code}")
        if 200 <= resp.status_code < 300:
            data = resp.json()
            typer.echo(
                "auth OK — account_number="
                f"{data.get('account_number')} status={data.get('status')} "
                f"cash={data.get('cash')} equity={data.get('equity')}"
            )
            return

        typer.echo(f"response body    : {resp.text[:500]}", err=True)
        if resp.status_code == 401:
            typer.echo(
                "\nLikely causes:\n"
                "  - Key/secret swapped in .env\n"
                "  - Keys from a different environment than ALPACA_BASE_URL\n"
                "    (paper keys -> paper-api.alpaca.markets; live -> api.alpaca.markets)\n"
                "  - Key was regenerated; old secret no longer valid\n"
                "  - Secret truncated when copied (Alpaca only shows it once)\n"
                "  - Account is Broker API / OAuth, not regular Trading API\n"
                "Regenerate at: https://app.alpaca.markets/paper/dashboard/overview",
                err=True,
            )
        raise typer.Exit(code=1)

    # silence unused-name warnings; typer registers via decorator side effects
    _ = (reconcile_cmd, check_alpaca_cmd)
    _ = AlpacaError  # re-exported for callers who catch it
