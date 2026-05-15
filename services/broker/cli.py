"""Phase 3 broker CLI subcommands."""
from __future__ import annotations

import typer

from packages.shared.config import get_settings
from packages.shared.db import session_scope
from services.broker.alpaca import AlpacaClient
from services.broker.reconcile import reconcile_once


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
