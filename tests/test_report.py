"""`stockpred report` CLI smoke tests."""
from __future__ import annotations

import datetime as dt

from typer.testing import CliRunner

from packages.shared.models import Position, RiskState, Signal, Ticker
from services.ingestion.cli import app

runner = CliRunner()


def test_report_with_empty_db_does_not_crash(session):  # pylint: disable=unused-argument
    # `session` fixture monkeypatches the shared db sessionmaker → the CLI uses the test DB.
    result = runner.invoke(app, ["report"])
    assert result.exit_code == 0, result.output
    assert "Positions" in result.output
    assert "Latest signals" in result.output
    assert "Risk state" in result.output


def test_report_renders_populated_db(session):
    session.add(Ticker(symbol="AAPL", active=True))
    session.flush()
    session.add(
        Position(
            symbol="AAPL", qty=10, avg_price=150.0, source="alpaca",
            updated_at=dt.datetime.now(dt.timezone.utc),
        )
    )
    session.add(
        Signal(
            symbol="AAPL",
            ts=dt.datetime(2024, 5, 1, 16, tzinfo=dt.timezone.utc),
            model_version="v1",
            score=0.7,
            decision="BUY",
            rationale={"reason": "all_gates_passed"},
        )
    )
    session.add(
        RiskState(
            ts=dt.datetime.now(dt.timezone.utc),
            equity=100_000.0,
            cash=50_000.0,
            exposure_pct=50.0,
            n_open_positions=1,
            halted=False,
        )
    )
    session.commit()

    result = runner.invoke(app, ["report"])
    assert result.exit_code == 0, result.output
    assert "AAPL" in result.output
    assert "BUY" in result.output
    assert "equity=100000.00" in result.output
