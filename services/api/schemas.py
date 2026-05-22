"""Pydantic response models for the dashboard API."""
from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, ConfigDict


class _ApiModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class PositionOut(_ApiModel):
    symbol: str
    qty: float
    avg_price: float
    updated_at: dt.datetime
    source: str


class SignalOut(_ApiModel):
    id: int
    symbol: str
    ts: dt.datetime
    model_version: str
    score: float
    decision: str
    rationale: dict[str, Any]
    created_at: dt.datetime


class OrderOut(_ApiModel):
    id: int
    symbol: str
    side: str
    qty: float
    order_type: str
    limit_price: float | None
    stop_price: float | None
    take_profit: float | None
    status: str
    submitted_at: dt.datetime
    filled_at: dt.datetime | None
    broker_order_id: str | None
    signal_id: int | None


class EquityPoint(_ApiModel):
    ts: dt.datetime
    equity: float
    cash: float
    exposure_pct: float
    max_drawdown: float
    n_open_positions: int
    halted: bool


class HealthOut(BaseModel):
    status: str = "ok"
