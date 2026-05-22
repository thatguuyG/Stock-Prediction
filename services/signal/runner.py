"""Signal runner: load latest predictions+features+positions → decide → write signals+orders."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from packages.shared.config import get_settings
from packages.shared.db import upsert_ignore
from packages.shared.logging import get_logger
from packages.shared.models import Order, Position, Prediction, PriceBar, RiskState, Signal
from services.broker.alpaca import AlpacaClient, AlpacaError
from services.model.features import build_feature_matrix
from services.signal.rules import (
    DEFAULT_MAX_EXPOSURE_PCT,
    DEFAULT_MAX_POSITIONS,
    DEFAULT_SENTIMENT_FLOOR,
    DEFAULT_TARGET_PCT,
    DEFAULT_THRESHOLD,
    SignalDecision,
    decide,
)

log = get_logger(__name__)

STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.04


@dataclass
class RunSummary:
    halted: bool = False
    n_signals: int = 0
    n_buy: int = 0
    n_sell: int = 0
    n_hold: int = 0
    n_orders_submitted: int = 0
    n_orders_dry_run: int = 0
    errors: list[str] = field(default_factory=list)


def _latest_predictions(session: Session, model_version: str) -> pd.DataFrame:
    rows = session.execute(
        select(Prediction.symbol, Prediction.ts, Prediction.score)
        .where(Prediction.model_version == model_version)
        .order_by(Prediction.symbol, Prediction.ts.desc())
    ).all()
    if not rows:
        return pd.DataFrame(columns=["symbol", "ts", "score"])
    df = pd.DataFrame(rows, columns=["symbol", "ts", "score"])
    return df.groupby("symbol", as_index=False).first()


def _latest_features(session: Session) -> pd.DataFrame:
    df = build_feature_matrix(session)
    if df.empty:
        return df
    latest = df.sort_values("ts").groupby("symbol", as_index=False).tail(1)
    closes = pd.DataFrame(
        session.execute(select(PriceBar.symbol, PriceBar.ts, PriceBar.close)).all(),
        columns=["symbol", "ts", "close"],
    )
    if closes.empty:
        latest = latest.copy()
        latest["close"] = float("nan")
        return latest
    closes["ts"] = pd.to_datetime(closes["ts"], utc=True).dt.tz_localize(None)
    latest = latest.copy()
    latest["ts"] = pd.to_datetime(latest["ts"], utc=True).dt.tz_localize(None)
    return latest.merge(closes, on=["symbol", "ts"], how="left")


def _portfolio_state(session: Session) -> tuple[float, float, int, dict[str, float]]:
    """Returns (equity, current_exposure_pct, n_open_positions, qty_by_symbol)."""
    rs = session.execute(
        select(RiskState).order_by(RiskState.ts.desc()).limit(1)
    ).scalar_one_or_none()
    equity = float(rs.equity) if rs is not None else 0.0
    exposure_pct = float(rs.exposure_pct) if rs is not None else 0.0

    rows = session.execute(
        select(Position.symbol, Position.qty).where(Position.qty != 0)
    ).all()
    qty_by_symbol = {sym: float(qty) for sym, qty in rows}
    n_open = len(qty_by_symbol)
    return equity, exposure_pct, n_open, qty_by_symbol


def _qty_for_target(target_pct: float, equity: float, price: float) -> int:
    if equity <= 0 or price <= 0:
        return 0
    notional = equity * (target_pct / 100.0)
    return max(int(notional // price), 0)


def _persist_signal(
    session: Session,
    model_version: str,
    ts: dt.datetime,
    score: float,
    decision: SignalDecision,
) -> Signal | None:
    inserted = upsert_ignore(
        session,
        Signal.__table__,
        [
            {
                "symbol": decision.symbol,
                "ts": ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                "model_version": model_version,
                "score": float(score),
                "decision": decision.decision,
                "rationale": decision.rationale,
            }
        ],
        ["symbol", "ts", "model_version"],
    )
    if inserted == 0:
        return None
    return session.execute(
        select(Signal)
        .where(Signal.symbol == decision.symbol)
        .where(Signal.ts == ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts)
        .where(Signal.model_version == model_version)
    ).scalar_one_or_none()


def _submit_order(
    session: Session,
    signal_row: Signal,
    decision: SignalDecision,
    close: float,
    equity: float,
    alpaca: AlpacaClient | None,
    summary: RunSummary,
) -> None:
    side = "buy" if decision.decision == "BUY" else "sell"
    qty = _qty_for_target(decision.target_pct, equity, close)
    if qty <= 0:
        log.info(
            "%s %s: target qty <= 0 (equity=%.2f close=%.2f), skipping",
            side, decision.symbol, equity, close,
        )
        return

    take_profit = close * (1.0 + TAKE_PROFIT_PCT) if side == "buy" else close * (1.0 - TAKE_PROFIT_PCT)
    stop_loss = close * (1.0 - STOP_LOSS_PCT) if side == "buy" else close * (1.0 + STOP_LOSS_PCT)

    order = Order(
        symbol=decision.symbol,
        side=side,
        qty=qty,
        order_type="market",
        take_profit=take_profit,
        stop_price=stop_loss,
        status="pending",
        signal_id=signal_row.id,
    )
    session.add(order)
    session.flush()

    if alpaca is None:
        order.status = "dry_run"
        summary.n_orders_dry_run += 1
        log.info("DRY RUN order: %s %s qty=%d", side, decision.symbol, qty)
        return

    try:
        resp = alpaca.submit_bracket_order(
            symbol=decision.symbol,
            qty=qty,
            side=side,
            take_profit=take_profit,
            stop_loss=stop_loss,
        )
        order.broker_order_id = resp.get("id")
        order.status = resp.get("status", "submitted")
        summary.n_orders_submitted += 1
        log.info(
            "Submitted %s %s qty=%d broker_id=%s",
            side, decision.symbol, qty, order.broker_order_id,
        )
    except AlpacaError as exc:
        order.status = "rejected"
        summary.errors.append(f"{decision.symbol}: {exc}")
        log.error("Alpaca submission failed for %s: %s", decision.symbol, exc)


def run_once(
    session: Session,
    model_version: str,
    *,
    alpaca: AlpacaClient | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    sentiment_floor: float = DEFAULT_SENTIMENT_FLOOR,
    target_pct: float = DEFAULT_TARGET_PCT,
    max_positions: int = DEFAULT_MAX_POSITIONS,
    max_exposure_pct: float = DEFAULT_MAX_EXPOSURE_PCT,
) -> RunSummary:
    summary = RunSummary()

    if get_settings().risk_halt:
        log.warning("RISK_HALT is set — no signals will be processed")
        summary.halted = True
        return summary

    preds = _latest_predictions(session, model_version)
    if preds.empty:
        log.warning("run_once: no predictions found for model_version=%s", model_version)
        return summary

    feats = _latest_features(session)
    if feats.empty:
        log.warning("run_once: no feature rows available")
        return summary

    # Normalize ts to tz-naive on both sides so merges work across dialects.
    preds = preds.copy()
    feats = feats.copy()
    preds["ts"] = pd.to_datetime(preds["ts"], utc=True).dt.tz_localize(None)
    feats["ts"] = pd.to_datetime(feats["ts"], utc=True).dt.tz_localize(None)

    merged = preds.merge(
        feats[["symbol", "ts", "close", "sma_50", "sent_mean_7d"]],
        on=["symbol", "ts"],
        how="inner",
    )
    if merged.empty:
        log.warning(
            "run_once: no (symbol, ts) match between predictions and features for %s",
            model_version,
        )
        return summary

    equity, exposure_pct, n_open, qty_by_symbol = _portfolio_state(session)

    for row in merged.itertuples(index=False):
        decision = decide(
            symbol=str(row.symbol),
            score=float(row.score),
            close=float(row.close),
            sma_50=float(row.sma_50),
            sentiment_7d=float(row.sent_mean_7d),
            current_position_qty=qty_by_symbol.get(str(row.symbol), 0.0),
            current_exposure_pct=exposure_pct,
            n_open_positions=n_open,
            threshold=threshold,
            sentiment_floor=sentiment_floor,
            target_pct=target_pct,
            max_positions=max_positions,
            max_exposure_pct=max_exposure_pct,
        )
        summary.n_signals += 1
        if decision.decision == "BUY":
            summary.n_buy += 1
        elif decision.decision == "SELL":
            summary.n_sell += 1
        else:
            summary.n_hold += 1

        sig = _persist_signal(session, model_version, row.ts, float(row.score), decision)
        session.flush()
        if sig is None:
            continue
        if decision.decision in ("BUY", "SELL"):
            _submit_order(session, sig, decision, float(row.close), equity, alpaca, summary)
            session.flush()

    return summary
