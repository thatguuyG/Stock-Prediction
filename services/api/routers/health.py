"""Liveness probe."""
from __future__ import annotations

from fastapi import APIRouter

from services.api.schemas import HealthOut

router = APIRouter()


@router.get("/healthz", response_model=HealthOut)
def healthz() -> HealthOut:
    return HealthOut(status="ok")
