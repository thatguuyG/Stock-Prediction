"""GET /equity tests."""
from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient

from packages.shared.models import RiskState
from services.api.deps import get_session
from services.api.main import app


@pytest.fixture(name="client")
def fixture_client(session):
    app.dependency_overrides[get_session] = lambda: session
    yield TestClient(app)
    app.dependency_overrides.clear()


def _seed(session, n: int = 5):
    for i in range(n):
        session.add(
            RiskState(
                ts=dt.datetime(2024, 5, 1 + i, tzinfo=dt.timezone.utc),
                equity=100_000.0 + i * 1_000,
                cash=50_000.0,
                exposure_pct=50.0,
                n_open_positions=2,
                halted=False,
            )
        )
    session.commit()


def test_returns_in_ascending_order(client, session):
    _seed(session, 3)
    resp = client.get("/equity")
    data = resp.json()
    assert [d["equity"] for d in data] == [100_000.0, 101_000.0, 102_000.0]


def test_empty_db_returns_empty_list(client):
    resp = client.get("/equity")
    assert resp.json() == []


def test_date_range_filter(client, session):
    _seed(session, 5)
    resp = client.get("/equity", params={"from": "2024-05-02", "to": "2024-05-04"})
    data = resp.json()
    assert len(data) == 3
    assert [d["equity"] for d in data] == [101_000.0, 102_000.0, 103_000.0]


def test_response_shape(client, session):
    _seed(session, 1)
    row = client.get("/equity").json()[0]
    assert set(row.keys()) == {
        "ts", "equity", "cash", "exposure_pct",
        "max_drawdown", "n_open_positions", "halted",
    }
