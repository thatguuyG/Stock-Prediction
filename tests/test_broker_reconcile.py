"""Reconciler tests — synthetic broker state, assert local table mutations."""
from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock

from sqlalchemy import select

from packages.shared.models import Order, Position, RiskState, Ticker, Trade
from services.broker.reconcile import reconcile_once


def _seed_ticker_and_order(session, broker_order_id: str = "ord-1") -> Order:
    session.add(Ticker(symbol="AAPL", active=True))
    session.flush()
    o = Order(
        symbol="AAPL",
        side="buy",
        qty=5,
        order_type="market",
        status="submitted",
        broker_order_id=broker_order_id,
    )
    session.add(o)
    session.flush()
    return o


def _mock_alpaca(
    *,
    order_payload=None,
    positions=None,
    account=None,
) -> MagicMock:
    m = MagicMock()
    m.get_order.return_value = order_payload or {}
    m.list_positions.return_value = positions or []
    m.get_account.return_value = account or {
        "equity": "100000",
        "cash": "100000",
        "long_market_value": "0",
    }
    return m


def test_open_order_status_updates(session):
    _seed_ticker_and_order(session)
    alpaca = _mock_alpaca(
        order_payload={
            "status": "filled",
            "filled_qty": "5",
            "filled_avg_price": "150.0",
            "filled_at": "2024-05-01T15:00:00Z",
        },
    )
    summary = reconcile_once(session, alpaca)
    session.flush()
    assert summary.n_orders_updated == 1
    assert summary.n_new_trades == 1
    order = session.execute(select(Order)).scalar_one()
    assert order.status == "filled"
    assert order.filled_at is not None


def test_fill_creates_trade_row(session):
    _seed_ticker_and_order(session)
    alpaca = _mock_alpaca(
        order_payload={"status": "filled", "filled_qty": "5", "filled_avg_price": "151.50"},
    )
    reconcile_once(session, alpaca)
    session.flush()
    trades = session.execute(select(Trade)).scalars().all()
    assert len(trades) == 1
    assert trades[0].price == 151.50
    assert trades[0].qty == 5


def test_partial_fill_then_full_fill_inserts_only_delta(session):
    _seed_ticker_and_order(session)
    alpaca_partial = _mock_alpaca(
        order_payload={"status": "partially_filled", "filled_qty": "2", "filled_avg_price": "150.0"}
    )
    reconcile_once(session, alpaca_partial)
    session.flush()
    alpaca_full = _mock_alpaca(
        order_payload={"status": "filled", "filled_qty": "5", "filled_avg_price": "150.5"}
    )
    reconcile_once(session, alpaca_full)
    session.flush()
    trades = session.execute(select(Trade)).scalars().all()
    assert len(trades) == 2
    qtys = sorted(t.qty for t in trades)
    assert qtys == [2.0, 3.0]


def test_positions_mirrored_from_broker(session):
    session.add(Ticker(symbol="AAPL", active=True))
    session.add(Ticker(symbol="MSFT", active=True))
    session.flush()
    alpaca = _mock_alpaca(
        positions=[
            {"symbol": "AAPL", "qty": "10", "avg_entry_price": "150.0"},
            {"symbol": "MSFT", "qty": "5", "avg_entry_price": "300.0"},
        ],
    )
    reconcile_once(session, alpaca)
    session.flush()
    positions = session.execute(select(Position)).scalars().all()
    assert {p.symbol for p in positions} == {"AAPL", "MSFT"}
    aapl = session.get(Position, "AAPL")
    assert aapl.qty == 10.0
    assert aapl.avg_price == 150.0


def test_position_zeroed_when_broker_no_longer_holds(session):
    session.add(Ticker(symbol="AAPL", active=True))
    session.add(
        Position(
            symbol="AAPL", qty=10, avg_price=150.0, source="alpaca",
            updated_at=dt.datetime.now(dt.timezone.utc),
        )
    )
    session.flush()
    alpaca = _mock_alpaca(positions=[])  # Alpaca says we hold nothing
    reconcile_once(session, alpaca)
    session.flush()
    aapl = session.get(Position, "AAPL")
    assert aapl.qty == 0.0


def test_risk_state_row_written(session):
    alpaca = _mock_alpaca(
        account={"equity": "120000", "cash": "60000", "long_market_value": "60000"},
    )
    summary = reconcile_once(session, alpaca)
    session.flush()
    assert summary.equity == 120000.0
    assert summary.exposure_pct == 50.0
    rs_rows = session.execute(select(RiskState)).scalars().all()
    assert len(rs_rows) == 1
    assert rs_rows[0].equity == 120000.0
    assert rs_rows[0].exposure_pct == 50.0
