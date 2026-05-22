"""GET /equity — risk_state time series for the equity-curve chart."""
from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from packages.shared.models import RiskState
from services.api.deps import get_session
from services.api.schemas import EquityPoint

router = APIRouter()


@router.get("/equity", response_model=list[EquityPoint])
def equity_curve(
    from_: dt.date | None = Query(default=None, alias="from"),
    to: dt.date | None = None,
    session: Session = Depends(get_session),
) -> list[EquityPoint]:
    stmt = select(RiskState).order_by(RiskState.ts.asc())
    if from_ is not None:
        stmt = stmt.where(RiskState.ts >= dt.datetime.combine(from_, dt.time.min))
    if to is not None:
        stmt = stmt.where(RiskState.ts <= dt.datetime.combine(to, dt.time.max))
    rows = session.execute(stmt).scalars().all()
    return [EquityPoint.model_validate(r) for r in rows]
