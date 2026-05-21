"""GET /positions tests."""
from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient

from packages.shared.models import Position, Ticker
from services.api.deps import get_session
from services.api.main import app


@pytest.fixture(name="client")
def fixture_client(session):
    app.dependency_overrides[get_session] = lambda: session
    yield TestClient(app)
    app.dependency_overrides.clear()


def _now() -> dt.datetime:
    return dt.datetime(2024, 5, 1, tzinfo=dt.timezone.utc)


def test_returns_non_zero_positions_only(client, session):
    session.add_all([Ticker(symbol=s, active=True) for s in ("AAPL", "MSFT", "NVDA")])
    session.flush()
    session.add_all(
        [
            Position(symbol="AAPL", qty=10, avg_price=150.0, source="alpaca", updated_at=_now()),
            Position(symbol="MSFT", qty=0, avg_price=0.0, source="alpaca", updated_at=_now()),
            Position(symbol="NVDA", qty=5, avg_price=400.0, source="alpaca", updated_at=_now()),
        ]
    )
    session.commit()

    resp = client.get("/positions")
    assert resp.status_code == 200
    data = resp.json()
    assert {p["symbol"] for p in data} == {"AAPL", "NVDA"}
    assert all(p["qty"] != 0 for p in data)


def test_empty_db_returns_empty_list(client):
    resp = client.get("/positions")
    assert resp.status_code == 200
    assert resp.json() == []


def test_response_shape(client, session):
    session.add(Ticker(symbol="AAPL", active=True))
    session.flush()
    session.add(
        Position(symbol="AAPL", qty=10, avg_price=150.0, source="alpaca", updated_at=_now())
    )
    session.commit()

    resp = client.get("/positions")
    row = resp.json()[0]
    assert set(row.keys()) == {"symbol", "qty", "avg_price", "updated_at", "source"}
