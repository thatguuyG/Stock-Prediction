"""Typer CLI for Phase 1 ingestion."""
from __future__ import annotations

import datetime as dt

import typer

from packages.shared.config import get_settings
from packages.shared.db import session_scope
from packages.shared.logging import get_logger
from services.ingestion import features, news, prices, sentiment
from services.model import cli as model_cli

app = typer.Typer(add_completion=False, help="Stock-Prediction CLI — Phase 1 ingestion + Phase 2 model.")
log = get_logger(__name__)
model_cli.register(app)


def _watchlist() -> list[str]:
    wl = get_settings().watchlist
    if not wl:
        raise typer.BadParameter("WATCHLIST env var is empty.")
    return wl


@app.command("ingest-prices")
def ingest_prices(
    since: str = typer.Option("2020-01-01", help="ISO date — earliest bar to fetch."),
) -> None:
    start = dt.date.fromisoformat(since)
    with session_scope() as session:
        for symbol in _watchlist():
            prices.ingest_symbol(session, symbol, start)


@app.command("compute-features")
def compute_features() -> None:
    with session_scope() as session:
        for symbol in _watchlist():
            features.compute_for_symbol(session, symbol)


@app.command("ingest-news")
def ingest_news() -> None:
    with session_scope() as session:
        for symbol in _watchlist():
            news.ingest_symbol(session, symbol)


@app.command("score-sentiment")
def score_sentiment() -> None:
    with session_scope() as session:
        sentiment.score_unscored(session)


@app.command("run-daily")
def run_daily(
    since: str = typer.Option("2020-01-01", help="ISO date — earliest bar to fetch."),
) -> None:
    """Run all four ingestion steps in sequence. Cron entry point."""
    ingest_prices(since=since)
    compute_features()
    ingest_news()
    score_sentiment()


if __name__ == "__main__":
    app()
