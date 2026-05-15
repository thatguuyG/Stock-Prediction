"""Signal decision logic — pure function, no I/O.

The decision pipeline produces a BUY / SELL / HOLD decision together with
a full rationale dict that captures every rule's input and whether it
passed. The rationale is what gets persisted to `signals.rationale` and
queried later for audit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Decision = Literal["BUY", "SELL", "HOLD"]

DEFAULT_THRESHOLD = 0.55
DEFAULT_SENTIMENT_FLOOR = -0.1
DEFAULT_TARGET_PCT = 5.0
DEFAULT_MAX_POSITIONS = 20
DEFAULT_MAX_EXPOSURE_PCT = 80.0


@dataclass
class SignalDecision:
    symbol: str
    decision: Decision
    target_pct: float
    rationale: dict[str, Any]


def decide(  # pylint: disable=too-many-return-statements
    *,
    symbol: str,
    score: float,
    close: float,
    sma_50: float,
    sentiment_7d: float,
    current_position_qty: float,
    current_exposure_pct: float,
    n_open_positions: int,
    threshold: float = DEFAULT_THRESHOLD,
    sentiment_floor: float = DEFAULT_SENTIMENT_FLOOR,
    target_pct: float = DEFAULT_TARGET_PCT,
    max_positions: int = DEFAULT_MAX_POSITIONS,
    max_exposure_pct: float = DEFAULT_MAX_EXPOSURE_PCT,
) -> SignalDecision:
    """Apply the long-only Phase 3 rule pipeline.

    BUY  — bullish score + trend + benign sentiment + portfolio capacity, no current position.
    SELL — bearish score (score < 1 - threshold) while already long.
    HOLD — everything else (including the no-op of being long with no exit trigger).
    """
    already_long = current_position_qty > 0
    bullish = score > threshold
    bearish = score < (1.0 - threshold)
    trend_ok = close > sma_50
    sentiment_ok = sentiment_7d > sentiment_floor
    has_position_slot = n_open_positions < max_positions
    exposure_ok = (current_exposure_pct + target_pct) <= max_exposure_pct

    rationale: dict[str, Any] = {
        "score": score,
        "threshold": threshold,
        "bullish": bullish,
        "bearish": bearish,
        "close": close,
        "sma_50": sma_50,
        "trend_ok": trend_ok,
        "sentiment_7d": sentiment_7d,
        "sentiment_floor": sentiment_floor,
        "sentiment_ok": sentiment_ok,
        "n_open_positions": n_open_positions,
        "max_positions": max_positions,
        "has_position_slot": has_position_slot,
        "current_exposure_pct": current_exposure_pct,
        "target_pct": target_pct,
        "max_exposure_pct": max_exposure_pct,
        "exposure_ok": exposure_ok,
        "already_long": already_long,
    }

    if already_long and bearish:
        rationale["reason"] = "exit_on_bearish_score"
        return SignalDecision(symbol=symbol, decision="SELL", target_pct=0.0, rationale=rationale)

    if already_long:
        rationale["reason"] = "hold_existing_position"
        return SignalDecision(symbol=symbol, decision="HOLD", target_pct=0.0, rationale=rationale)

    if not bullish:
        rationale["reason"] = "score_below_threshold"
        return SignalDecision(symbol=symbol, decision="HOLD", target_pct=0.0, rationale=rationale)

    if not trend_ok:
        rationale["reason"] = "trend_filter_failed"
        return SignalDecision(symbol=symbol, decision="HOLD", target_pct=0.0, rationale=rationale)

    if not sentiment_ok:
        rationale["reason"] = "sentiment_gate_failed"
        return SignalDecision(symbol=symbol, decision="HOLD", target_pct=0.0, rationale=rationale)

    if not has_position_slot:
        rationale["reason"] = "max_positions_reached"
        return SignalDecision(symbol=symbol, decision="HOLD", target_pct=0.0, rationale=rationale)

    if not exposure_ok:
        rationale["reason"] = "exposure_cap_reached"
        return SignalDecision(symbol=symbol, decision="HOLD", target_pct=0.0, rationale=rationale)

    rationale["reason"] = "all_gates_passed"
    return SignalDecision(symbol=symbol, decision="BUY", target_pct=target_pct, rationale=rationale)
