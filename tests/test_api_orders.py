"""GET /orders tests."""
from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient

from packages.shared.models import Order, Ticker
from services.api.deps import get_session
from services.api.main import app


@pytest.fixture(name="client")
def fixture_client(session):
    app.dependency_overrides[get_session] = lambda: session
    yield TestClient(app)
    app.dependency_overrides.clear()


def _seed(session, *statuses):
    session.add(Ticker(symbol="AAPL", active=True))
    session.flush()
    for i, st in enumerate(statuses):
        session.add(
            Order(
                symbol="AAPL",
                side="buy",
                qty=5,
                order_type="market",
                status=st,
                submitted_at=dt.datetime(2024, 5, 1 + i, tzinfo=dt.timezone.utc),
            )
        )
    session.commit()


def test_returns_most_recent_first(client, session):
    _seed(session, "pending", "filled", "rejected")
    resp = client.get("/orders")
    data = resp.json()
    assert [o["status"] for o in data] == ["rejected", "filled", "pending"]


def test_filter_by_status(client, session):
    _seed(session, "pending", "filled", "filled")
    resp = client.get("/orders", params={"status": "filled"})
    data = resp.json()
    assert len(data) == 2
    assert all(o["status"] == "filled" for o in data)


def test_limit_caps_results(client, session):
    _seed(session, *["pending"] * 10)
    resp = client.get("/orders", params={"limit": 3})
    assert len(resp.json()) == 3


def test_empty_db_returns_empty_list(client):
    resp = client.get("/orders")
    assert resp.json() == []
