"""Liveness probe smoke test."""
from __future__ import annotations

from fastapi.testclient import TestClient

from services.api.main import app


def test_healthz():
    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
