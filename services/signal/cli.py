"""Phase 3 signal-engine CLI subcommands."""
from __future__ import annotations

import typer
from sqlalchemy import select

from packages.shared.config import get_settings
from packages.shared.db import session_scope
from packages.shared.models import Position, RiskState, Signal
from services.broker.alpaca import AlpacaClient
from services.signal.runner import run_once

DEFAULT_REPORT_LIMIT = 50


def register(app: typer.Typer) -> None:
    """Mount signal + report commands onto the top-level stockpred app."""

    @app.command("run-signals")
    def run_signals_cmd(
        model_version: str = typer.Option(..., "--model-version"),
        dry_run: bool = typer.Option(
            False, "--dry-run", help="Skip Alpaca submission; mark orders dry_run."
        ),
    ) -> None:
        settings = get_settings()
        alpaca: AlpacaClient | None = None
        if not dry_run and settings.alpaca_key_id and settings.alpaca_secret_key:
            alpaca = AlpacaClient()
        elif not dry_run:
            typer.echo("WARNING: ALPACA_KEY_ID/SECRET not set; running in dry-run mode.", err=True)

        with session_scope() as session:
            summary = run_once(session, model_version=model_version, alpaca=alpaca)
        typer.echo(
            f"run-signals: halted={summary.halted} signals={summary.n_signals} "
            f"(buy={summary.n_buy} sell={summary.n_sell} hold={summary.n_hold}) "
            f"submitted={summary.n_orders_submitted} dry_run={summary.n_orders_dry_run}"
        )
        if summary.errors:
            for err in summary.errors:
                typer.echo(f"  error: {err}", err=True)

    @app.command("report")
    def report_cmd(
        limit: int = typer.Option(DEFAULT_REPORT_LIMIT, help="Max recent signals to display."),
    ) -> None:
        with session_scope() as session:
            positions = session.execute(
                select(Position).where(Position.qty != 0).order_by(Position.symbol)
            ).scalars().all()
            recent_signals = session.execute(
                select(Signal).order_by(Signal.ts.desc()).limit(limit)
            ).scalars().all()
            latest_risk = session.execute(
                select(RiskState).order_by(RiskState.ts.desc()).limit(1)
            ).scalar_one_or_none()

        typer.echo("=== Positions ===")
        if not positions:
            typer.echo("  (none)")
        for p in positions:
            typer.echo(f"  {p.symbol:6s} qty={p.qty:>10.2f}  avg={p.avg_price:>8.2f}  src={p.source}")

        typer.echo(f"\n=== Latest signals (up to {limit}) ===")
        if not recent_signals:
            typer.echo("  (none)")
        for s in recent_signals:
            reason = s.rationale.get("reason", "?") if isinstance(s.rationale, dict) else "?"
            typer.echo(
                f"  {s.ts.date()} {s.symbol:6s} {s.decision:4s} "
                f"score={s.score:.3f}  reason={reason}"
            )

        typer.echo("\n=== Risk state ===")
        if latest_risk is None:
            typer.echo("  (no reconcile yet — run `stockpred reconcile`)")
        else:
            typer.echo(
                f"  ts={latest_risk.ts.isoformat()}  equity={latest_risk.equity:.2f}  "
                f"cash={latest_risk.cash:.2f}  exposure={latest_risk.exposure_pct:.2f}%  "
                f"open={latest_risk.n_open_positions}  halted={latest_risk.halted}"
            )
