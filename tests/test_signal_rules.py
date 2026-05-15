"""Signal rule unit tests — every gate's pass and fail path."""
from __future__ import annotations

import pytest

from services.signal.rules import decide


def _base_kwargs(**overrides):
    base = {
        "symbol": "AAPL",
        "score": 0.7,
        "close": 110.0,
        "sma_50": 100.0,
        "sentiment_7d": 0.1,
        "current_position_qty": 0.0,
        "current_exposure_pct": 10.0,
        "n_open_positions": 2,
    }
    base.update(overrides)
    return base


def test_buy_when_all_gates_pass():
    d = decide(**_base_kwargs())
    assert d.decision == "BUY"
    assert d.target_pct == 5.0
    assert d.rationale["reason"] == "all_gates_passed"


def test_hold_when_score_below_threshold():
    d = decide(**_base_kwargs(score=0.4))
    assert d.decision == "HOLD"
    assert d.rationale["reason"] == "score_below_threshold"


def test_hold_when_trend_fails():
    d = decide(**_base_kwargs(close=90.0, sma_50=100.0))
    assert d.decision == "HOLD"
    assert d.rationale["reason"] == "trend_filter_failed"


def test_hold_when_sentiment_negative():
    d = decide(**_base_kwargs(sentiment_7d=-0.3))
    assert d.decision == "HOLD"
    assert d.rationale["reason"] == "sentiment_gate_failed"


def test_hold_when_max_positions_reached():
    d = decide(**_base_kwargs(n_open_positions=20))
    assert d.decision == "HOLD"
    assert d.rationale["reason"] == "max_positions_reached"


def test_hold_when_exposure_cap_reached():
    d = decide(**_base_kwargs(current_exposure_pct=78.0))
    # 78 + 5 = 83 > 80 cap
    assert d.decision == "HOLD"
    assert d.rationale["reason"] == "exposure_cap_reached"


def test_hold_when_already_long_and_neutral_score():
    d = decide(**_base_kwargs(current_position_qty=10.0, score=0.6))
    assert d.decision == "HOLD"
    assert d.rationale["reason"] == "hold_existing_position"


def test_sell_when_already_long_and_bearish_score():
    d = decide(**_base_kwargs(current_position_qty=10.0, score=0.3))
    assert d.decision == "SELL"
    assert d.rationale["reason"] == "exit_on_bearish_score"


def test_rationale_records_all_inputs():
    d = decide(**_base_kwargs())
    for key in [
        "score",
        "threshold",
        "close",
        "sma_50",
        "trend_ok",
        "sentiment_7d",
        "sentiment_ok",
        "n_open_positions",
        "current_exposure_pct",
        "target_pct",
        "exposure_ok",
        "reason",
    ]:
        assert key in d.rationale, f"missing rationale key: {key}"


@pytest.mark.parametrize(
    "score,expected_bullish,expected_bearish",
    [(0.7, True, False), (0.5, False, False), (0.3, False, True), (0.55, False, False)],
)
def test_score_classification_boundaries(score, expected_bullish, expected_bearish):
    d = decide(**_base_kwargs(score=score))
    assert d.rationale["bullish"] is expected_bullish
    assert d.rationale["bearish"] is expected_bearish
