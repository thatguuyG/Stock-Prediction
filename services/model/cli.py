"""Phase 2 model CLI subcommands: train / predict / backtest."""
from __future__ import annotations

import datetime as dt

import typer

from packages.shared.db import session_scope
from services.model.backtest import (
    DEFAULT_MAX_WEIGHT,
    DEFAULT_SLIPPAGE_BPS,
    DEFAULT_THRESHOLD,
    run_backtest,
)
from services.model.predict import predict as predict_fn
from services.model.train import train as train_fn


def register(app: typer.Typer) -> None:
    """Mount model commands onto the top-level stockpred app."""

    @app.command("train")
    def train_cmd(
        model_version: str = typer.Option(..., "--model-version", help="Free-form label, e.g. v1"),
        since: str | None = typer.Option(None, help="ISO date; restrict training to bars >= this."),
        train_window: int = typer.Option(252, help="Walk-forward training window (trading days)."),
        val_window: int = typer.Option(63, help="Walk-forward validation window (trading days)."),
        step: int = typer.Option(63, help="Walk-forward step size (trading days)."),
    ) -> None:
        since_date = dt.date.fromisoformat(since) if since else None
        with session_scope() as session:
            result = train_fn(
                session,
                model_version=model_version,
                since=since_date,
                train_window=train_window,
                val_window=val_window,
                step=step,
            )
        typer.echo(
            f"trained {model_version}: {result.n_folds} folds, "
            f"mean_acc={result.mean_accuracy:.3f}, mean_logloss={result.mean_log_loss:.3f}, "
            f"artifact={result.artifact_path}"
        )

    @app.command("predict")
    def predict_cmd(
        model_version: str = typer.Option(..., "--model-version"),
        since: str | None = typer.Option(None, help="ISO date; restrict inference to bars >= this."),
    ) -> None:
        since_date = dt.date.fromisoformat(since) if since else None
        with session_scope() as session:
            n = predict_fn(session, model_version=model_version, since=since_date)
        typer.echo(f"wrote {n} predictions for {model_version}")

    @app.command("backtest")
    def backtest_cmd(
        model_version: str = typer.Option(..., "--model-version"),
        threshold: float = typer.Option(DEFAULT_THRESHOLD),
        slippage_bps: float = typer.Option(DEFAULT_SLIPPAGE_BPS),
        max_weight: float = typer.Option(DEFAULT_MAX_WEIGHT),
        notes: str | None = typer.Option(None),
    ) -> None:
        with session_scope() as session:
            result = run_backtest(
                session,
                model_version=model_version,
                threshold=threshold,
                slippage_bps=slippage_bps,
                max_weight=max_weight,
                notes=notes,
            )
        typer.echo(
            f"backtest {model_version}: sharpe={result.sharpe:.3f} "
            f"total_return={result.total_return:.3f} max_dd={result.max_drawdown:.3f} "
            f"hit_rate={result.hit_rate:.3f} n_trades={result.n_trades}"
        )
