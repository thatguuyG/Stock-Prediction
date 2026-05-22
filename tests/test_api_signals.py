"""GET /signals tests."""
from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient

from packages.shared.models import Signal, Ticker
from services.api.deps import get_session
from services.api.main import app


@pytest.fixture(name="client")
def fixture_client(session):
    app.dependency_overrides[get_session] = lambda: session
    yield TestClient(app)
    app.dependency_overrides.clear()


def _seed(session, *decisions: str):
    session.add(Ticker(symbol="AAPL", active=True))
    session.flush()
    for i, dec in enumerate(decisions):
        session.add(
            Signal(
                symbol="AAPL",
                ts=dt.datetime(2024, 5, 1 + i, tzinfo=dt.timezone.utc),
                model_version="v1",
                score=0.7 if dec == "BUY" else 0.3 if dec == "SELL" else 0.5,
                decision=dec,
                rationale={"reason": f"r-{dec.lower()}", "score": 0.7},
            )
        )
    session.commit()


def test_returns_most_recent_first(client, session):
    _seed(session, "BUY", "HOLD", "SELL")
    resp = client.get("/signals")
    data = resp.json()
    assert [s["decision"] for s in data] == ["SELL", "HOLD", "BUY"]


def test_limit_param(client, session):
    _seed(session, "BUY", "HOLD", "SELL")
    resp = client.get("/signals", params={"limit": 2})
    assert len(resp.json()) == 2


def test_filter_by_decision(client, session):
    _seed(session, "BUY", "HOLD", "SELL", "BUY")
    resp = client.get("/signals", params={"decision": "BUY"})
    data = resp.json()
    assert len(data) == 2
    assert all(s["decision"] == "BUY" for s in data)


def test_rationale_roundtrip(client, session):
    _seed(session, "BUY")
    resp = client.get("/signals")
    assert resp.json()[0]["rationale"] == {"reason": "r-buy", "score": 0.7}


def test_invalid_decision_rejected(client):
    resp = client.get("/signals", params={"decision": "WAT"})
    assert resp.status_code == 422
