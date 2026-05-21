"""GET /orders — recent orders, optionally filtered by status."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from packages.shared.models import Order
from services.api.deps import get_session
from services.api.schemas import OrderOut

router = APIRouter()


@router.get("/orders", response_model=list[OrderOut])
def list_orders(
    limit: int = Query(50, ge=1, le=500),
    status: str | None = None,
    session: Session = Depends(get_session),
) -> list[OrderOut]:
    stmt = select(Order).order_by(Order.submitted_at.desc()).limit(limit)
    if status is not None:
        stmt = stmt.where(Order.status == status)
    rows = session.execute(stmt).scalars().all()
    return [OrderOut.model_validate(r) for r in rows]
