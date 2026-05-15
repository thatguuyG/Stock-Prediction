"""AlpacaClient tests — `responses` mocks every endpoint."""
from __future__ import annotations

import pytest
import responses

from services.broker.alpaca import AlpacaClient, AlpacaError

BASE = "https://test.example"


def _client() -> AlpacaClient:
    return AlpacaClient(
        key_id="k",
        secret_key="s",
        base_url=BASE,
        timeout=2,
        max_retries=2,
    )


def _auth_headers_ok(request) -> bool:
    return (
        request.headers.get("APCA-API-KEY-ID") == "k"
        and request.headers.get("APCA-API-SECRET-KEY") == "s"
    )


@responses.activate
def test_get_account_sends_auth_headers():
    responses.add(
        responses.GET,
        f"{BASE}/v2/account",
        json={"equity": "100000", "cash": "50000"},
        status=200,
    )
    out = _client().get_account()
    assert out == {"equity": "100000", "cash": "50000"}
    assert _auth_headers_ok(responses.calls[0].request)


@responses.activate
def test_list_positions_empty():
    responses.add(responses.GET, f"{BASE}/v2/positions", json=[], status=200)
    assert _client().list_positions() == []


@responses.activate
def test_submit_bracket_order_payload_shape():
    responses.add(
        responses.POST,
        f"{BASE}/v2/orders",
        json={"id": "abc-123", "status": "accepted"},
        status=200,
    )
    out = _client().submit_bracket_order(
        symbol="AAPL", qty=5, side="buy",
        take_profit=152.5, stop_loss=148.0,
    )
    assert out["id"] == "abc-123"
    req = responses.calls[0].request
    body = req.body.decode() if isinstance(req.body, bytes) else req.body
    assert '"symbol": "AAPL"' in body
    assert '"qty": 5' in body
    assert '"side": "buy"' in body
    assert '"order_class": "bracket"' in body
    assert '"limit_price": 152.5' in body
    assert '"stop_price": 148.0' in body


@responses.activate
def test_5xx_triggers_retry_then_succeeds():
    responses.add(responses.GET, f"{BASE}/v2/account", status=503)
    responses.add(responses.GET, f"{BASE}/v2/account", json={"equity": "1"}, status=200)
    assert _client().get_account() == {"equity": "1"}
    assert len(responses.calls) == 2


@responses.activate
def test_5xx_exhausts_retries_and_raises():
    responses.add(responses.GET, f"{BASE}/v2/account", status=500)
    responses.add(responses.GET, f"{BASE}/v2/account", status=500)
    with pytest.raises(AlpacaError):
        _client().get_account()
    assert len(responses.calls) == 2


@responses.activate
def test_4xx_raises_immediately():
    responses.add(responses.GET, f"{BASE}/v2/account", status=401, json={"message": "nope"})
    with pytest.raises(AlpacaError, match="401"):
        _client().get_account()
    assert len(responses.calls) == 1


@responses.activate
def test_get_order_uses_id_in_path():
    responses.add(
        responses.GET,
        f"{BASE}/v2/orders/abc-123",
        json={"id": "abc-123", "status": "filled", "filled_qty": "5", "filled_avg_price": "150.0"},
        status=200,
    )
    out = _client().get_order("abc-123")
    assert out["status"] == "filled"
