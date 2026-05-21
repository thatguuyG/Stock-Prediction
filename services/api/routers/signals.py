"""GET /signals — recent decisions, optionally filtered by decision."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from packages.shared.models import Signal
from services.api.deps import get_session
from services.api.schemas import SignalOut

router = APIRouter()


@router.get("/signals", response_model=list[SignalOut])
def list_signals(
    limit: int = Query(50, ge=1, le=500),
    decision: Literal["BUY", "SELL", "HOLD"] | None = None,
    session: Session = Depends(get_session),
) -> list[SignalOut]:
    stmt = select(Signal).order_by(Signal.ts.desc()).limit(limit)
    if decision is not None:
        stmt = stmt.where(Signal.decision == decision)
    rows = session.execute(stmt).scalars().all()
    return [SignalOut.model_validate(r) for r in rows]
