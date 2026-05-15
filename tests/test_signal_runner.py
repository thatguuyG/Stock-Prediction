"""Signal runner integration tests.

Stub the feature matrix to keep the test focused on signal/order writing
behaviour; the feature matrix itself is covered by tests/test_model_features.py.
"""
from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sqlalchemy import select

from packages.shared.config import get_settings
from packages.shared.models import Order, Position, Prediction, RiskState, Signal, Ticker
from services.signal import runner as runner_mod


def _seed_prediction(session, symbol: str = "AAPL", score: float = 0.7):
    session.add(Ticker(symbol=symbol, active=True))
    session.flush()
    ts = dt.datetime(2024, 5, 1, 16, 0, tzinfo=dt.timezone.utc)
    session.add(
        Prediction(
            symbol=symbol, ts=ts, model_version="v1",
            score=score, label_pred=int(score >= 0.5),
        )
    )
    session.flush()
    return symbol, ts


def _seed_risk_state(session, equity: float = 100_000.0):
    session.add(
        RiskState(
            ts=dt.datetime.now(dt.timezone.utc),
            equity=equity,
            cash=equity,
            exposure_pct=0.0,
            n_open_positions=0,
            halted=False,
        )
    )
    session.flush()


def _stub_features(monkeypatch, symbol: str, ts: dt.datetime, *, close: float, sma_50: float, sent: float):
    df = pd.DataFrame(
        [{"symbol": symbol, "ts": ts, "close": close, "sma_50": sma_50, "sent_mean_7d": sent}]
    )
    monkeypatch.setattr(runner_mod, "_latest_features", lambda _session: df)


@pytest.fixture(autouse=True)
def _reset_settings(monkeypatch):
    from packages.shared import config as cfg
    monkeypatch.setattr(cfg, "_settings", None)
    monkeypatch.setenv("RISK_HALT", "0")
    yield
    monkeypatch.setattr(cfg, "_settings", None)


def test_bullish_signal_creates_dry_run_order(session, monkeypatch):
    symbol, ts = _seed_prediction(session, score=0.7)
    _seed_risk_state(session, equity=100_000.0)
    _stub_features(monkeypatch, symbol, ts, close=110.0, sma_50=100.0, sent=0.1)

    summary = runner_mod.run_once(session, model_version="v1", alpaca=None)
    session.flush()

    assert summary.halted is False
    assert summary.n_buy == 1
    assert summary.n_orders_dry_run == 1
    signals = session.execute(select(Signal)).scalars().all()
    assert len(signals) == 1
    assert signals[0].decision == "BUY"
    orders = session.execute(select(Order)).scalars().all()
    assert len(orders) == 1
    assert orders[0].status == "dry_run"
    assert orders[0].side == "buy"
    assert orders[0].qty > 0
    assert orders[0].take_profit is not None
    assert orders[0].stop_price is not None


def test_bearish_score_with_no_position_holds(session, monkeypatch):
    symbol, ts = _seed_prediction(session, score=0.3)
    _seed_risk_state(session)
    _stub_features(monkeypatch, symbol, ts, close=110.0, sma_50=100.0, sent=0.1)

    summary = runner_mod.run_once(session, model_version="v1", alpaca=None)
    session.flush()

    assert summary.n_hold == 1
    assert summary.n_orders_dry_run == 0
    orders = session.execute(select(Order)).scalars().all()
    assert orders == []


def test_risk_halt_skips_all_processing(session, monkeypatch):
    from packages.shared import config as cfg
    _seed_prediction(session, score=0.9)
    _seed_risk_state(session)
    monkeypatch.setenv("RISK_HALT", "1")
    monkeypatch.setattr(cfg, "_settings", None)

    summary = runner_mod.run_once(session, model_version="v1", alpaca=None)
    session.flush()

    assert summary.halted is True
    assert summary.n_signals == 0
    assert session.execute(select(Signal)).all() == []
    assert session.execute(select(Order)).all() == []


def test_idempotent_rerun_no_duplicate_signals(session, monkeypatch):
    symbol, ts = _seed_prediction(session, score=0.7)
    _seed_risk_state(session)
    _stub_features(monkeypatch, symbol, ts, close=110.0, sma_50=100.0, sent=0.1)

    runner_mod.run_once(session, model_version="v1", alpaca=None)
    session.flush()
    runner_mod.run_once(session, model_version="v1", alpaca=None)
    session.flush()

    signals = session.execute(select(Signal)).scalars().all()
    assert len(signals) == 1


def test_alpaca_submission_records_broker_order_id(session, monkeypatch):
    symbol, ts = _seed_prediction(session, score=0.7)
    _seed_risk_state(session)
    _stub_features(monkeypatch, symbol, ts, close=110.0, sma_50=100.0, sent=0.1)

    alpaca = MagicMock()
    alpaca.submit_bracket_order.return_value = {"id": "br-123", "status": "accepted"}

    summary = runner_mod.run_once(session, model_version="v1", alpaca=alpaca)
    session.flush()

    assert summary.n_orders_submitted == 1
    order = session.execute(select(Order)).scalar_one()
    assert order.broker_order_id == "br-123"
    assert order.status == "accepted"


def test_existing_long_position_holds(session, monkeypatch):
    symbol, ts = _seed_prediction(session, score=0.7)
    _seed_risk_state(session)
    session.add(
        Position(symbol=symbol, qty=10.0, avg_price=100.0, source="alpaca",
                 updated_at=dt.datetime.now(dt.timezone.utc))
    )
    session.flush()
    _stub_features(monkeypatch, symbol, ts, close=110.0, sma_50=100.0, sent=0.1)

    summary = runner_mod.run_once(session, model_version="v1", alpaca=None)
    session.flush()

    assert summary.n_hold == 1
    assert session.execute(select(Order)).all() == []


def test_no_predictions_no_error(session):
    summary = runner_mod.run_once(session, model_version="v_missing", alpaca=None)
    assert summary.n_signals == 0
    assert summary.halted is False
