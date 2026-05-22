"""GET /positions — current portfolio (non-zero positions only)."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from packages.shared.models import Position
from services.api.deps import get_session
from services.api.schemas import PositionOut

router = APIRouter()


@router.get("/positions", response_model=list[PositionOut])
def list_positions(session: Session = Depends(get_session)) -> list[PositionOut]:
    rows = session.execute(
        select(Position).where(Position.qty != 0).order_by(Position.symbol)
    ).scalars().all()
    return [PositionOut.model_validate(r) for r in rows]
