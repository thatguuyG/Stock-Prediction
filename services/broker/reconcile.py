"""Reconcile local orders/positions/risk_state with Alpaca."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from packages.shared.logging import get_logger
from packages.shared.models import Order, Position, RiskState, Trade
from services.broker.alpaca import AlpacaClient, AlpacaError

log = get_logger(__name__)

TERMINAL_STATUSES = {"filled", "canceled", "rejected", "expired"}


@dataclass
class ReconcileSummary:
    n_orders_updated: int = 0
    n_new_trades: int = 0
    n_positions: int = 0
    equity: float = 0.0
    cash: float = 0.0
    exposure_pct: float = 0.0
    errors: list[str] = field(default_factory=list)


def _parse_alpaca_ts(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _sync_open_orders(session: Session, alpaca: AlpacaClient, summary: ReconcileSummary) -> None:
    open_orders = session.execute(
        select(Order)
        .where(Order.status.notin_(list(TERMINAL_STATUSES)))
        .where(Order.broker_order_id.is_not(None))
    ).scalars().all()

    for order in open_orders:
        try:
            broker_order = alpaca.get_order(order.broker_order_id)
        except AlpacaError as exc:
            summary.errors.append(f"order {order.broker_order_id}: {exc}")
            log.error("Failed to poll order %s: %s", order.broker_order_id, exc)
            continue

        new_status = broker_order.get("status") or order.status
        if new_status != order.status:
            order.status = new_status
            summary.n_orders_updated += 1

        filled_qty = float(broker_order.get("filled_qty") or 0.0)
        filled_avg_price = broker_order.get("filled_avg_price")
        filled_at = _parse_alpaca_ts(broker_order.get("filled_at"))

        if filled_qty > 0 and filled_avg_price is not None:
            existing_filled = sum(t.qty for t in order.trades)
            new_fill_qty = filled_qty - existing_filled
            if new_fill_qty > 0:
                session.add(
                    Trade(
                        order_id=order.id,
                        ts=filled_at or dt.datetime.now(dt.timezone.utc),
                        qty=new_fill_qty,
                        price=float(filled_avg_price),
                        fee=0.0,
                        broker_trade_id=f"{order.broker_order_id}:{int(filled_qty)}",
                    )
                )
                summary.n_new_trades += 1
        if new_status == "filled" and order.filled_at is None:
            order.filled_at = filled_at or dt.datetime.now(dt.timezone.utc)


def _sync_positions(session: Session, alpaca: AlpacaClient, summary: ReconcileSummary) -> None:
    broker_positions = alpaca.list_positions()
    seen_symbols: set[str] = set()

    for bp in broker_positions:
        symbol = bp.get("symbol")
        if not symbol:
            continue
        qty = float(bp.get("qty") or 0.0)
        avg_price = float(bp.get("avg_entry_price") or bp.get("cost_basis") or 0.0)
        existing = session.get(Position, symbol)
        if existing is None:
            session.add(
                Position(
                    symbol=symbol,
                    qty=qty,
                    avg_price=avg_price,
                    source="alpaca",
                    updated_at=dt.datetime.now(dt.timezone.utc),
                )
            )
        else:
            existing.qty = qty
            existing.avg_price = avg_price
            existing.source = "alpaca"
            existing.updated_at = dt.datetime.now(dt.timezone.utc)
        seen_symbols.add(symbol)
        summary.n_positions += 1

    locals_to_zero = session.execute(
        select(Position).where(Position.qty != 0).where(Position.source == "alpaca")
    ).scalars().all()
    for pos in locals_to_zero:
        if pos.symbol not in seen_symbols:
            pos.qty = 0.0
            pos.updated_at = dt.datetime.now(dt.timezone.utc)


def _write_risk_state(
    session: Session,
    alpaca: AlpacaClient,
    summary: ReconcileSummary,
    halted: bool = False,
) -> None:
    account = alpaca.get_account()
    equity = float(account.get("equity") or 0.0)
    cash = float(account.get("cash") or 0.0)
    long_value = float(account.get("long_market_value") or 0.0)
    exposure_pct = (long_value / equity * 100.0) if equity > 0 else 0.0

    n_open = session.execute(
        select(Position).where(Position.qty != 0)
    ).all()
    summary.equity = equity
    summary.cash = cash
    summary.exposure_pct = exposure_pct

    session.add(
        RiskState(
            ts=dt.datetime.now(dt.timezone.utc),
            equity=equity,
            cash=cash,
            exposure_pct=exposure_pct,
            n_open_positions=len(n_open),
            halted=halted,
        )
    )


def reconcile_once(session: Session, alpaca: AlpacaClient, *, halted: bool = False) -> ReconcileSummary:
    """Pull broker state into local Postgres. Writes orders/trades/positions/risk_state."""
    summary = ReconcileSummary()
    _sync_open_orders(session, alpaca, summary)
    session.flush()
    _sync_positions(session, alpaca, summary)
    session.flush()
    _write_risk_state(session, alpaca, summary, halted=halted)
    session.flush()
    log.info(
        "reconcile: orders_updated=%d new_trades=%d positions=%d equity=%.2f exposure=%.2f%%",
        summary.n_orders_updated,
        summary.n_new_trades,
        summary.n_positions,
        summary.equity,
        summary.exposure_pct,
    )
    return summary
