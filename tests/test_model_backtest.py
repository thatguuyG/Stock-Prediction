"""Backtester: hand-computed expectations on a tiny deterministic panel."""
from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import select

from packages.shared.models import BacktestRun, PriceBar, Prediction, Ticker
from services.model.backtest import run_backtest


def _insert_panel(session):
    """Two tickers, 6 days. A trends up monotonically and is always long-signal;
    B trends flat and is never long-signal."""
    base_ts = dt.datetime(2024, 1, 2, tzinfo=dt.timezone.utc)
    for sym in ("A", "B"):
        session.add(Ticker(symbol=sym, active=True))
    session.flush()

    a_closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    b_closes = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0]

    for i in range(6):
        ts = base_ts + dt.timedelta(days=i)
        for sym, closes in [("A", a_closes), ("B", b_closes)]:
            session.add(
                PriceBar(
                    symbol=sym,
                    ts=ts,
                    open=closes[i],
                    high=closes[i],
                    low=closes[i],
                    close=closes[i],
                    volume=1_000_000.0,
                    source="synth",
                )
            )
        session.add(
            Prediction(
                symbol="A", ts=ts, model_version="v1", score=1.0, label_pred=1,
            )
        )
        session.add(
            Prediction(
                symbol="B", ts=ts, model_version="v1", score=0.0, label_pred=0,
            )
        )
    session.flush()


def test_backtest_metrics_match_hand_computation(session):
    _insert_panel(session)
    result = run_backtest(
        session,
        model_version="v1",
        threshold=0.55,
        slippage_bps=5,
        max_weight=0.20,
    )
    session.flush()

    # Hand-computed gross daily returns (weight 0.20 on A, monotonically up):
    # day 0: 0.20 * 0.01 = 0.002 → net 0.0019 after 5bp × 0.20 = 0.0001 slippage
    # days 1..4: weighted next-day returns of A, no slippage
    # day 5: weight 0.20 but no next close → return 0
    expected_total = (
        (1 + 0.002 - 0.0001)
        * (1 + 0.20 * (1 / 101))
        * (1 + 0.20 * (1 / 102))
        * (1 + 0.20 * (1 / 103))
        * (1 + 0.20 * (1 / 104))
        - 1.0
    )
    assert result.total_return == pytest.approx(expected_total, abs=1e-6)
    # all positive-return days except the last (no next close), and day 0 nets 0.0019 > 0
    assert result.hit_rate == pytest.approx(5 / 6, abs=1e-6)
    # turnover = sum |Δweight| with the first row counted as |initial weight| = 0.20
    assert result.turnover == pytest.approx(0.20, abs=1e-9)
    assert result.n_days == 6
    assert result.n_symbols == 2


def test_backtest_writes_backtest_run_row(session):
    _insert_panel(session)
    run_backtest(session, model_version="v1")
    session.flush()
    rows = session.execute(
        select(BacktestRun).where(BacktestRun.model_version == "v1")
    ).all()
    assert len(rows) == 1
    row = rows[0][0]
    assert row.threshold == 0.55
    assert row.n_trades >= 1


def test_backtest_threshold_filters_below(session):
    _insert_panel(session)
    # threshold above A's score → no signals → all-zero returns
    result = run_backtest(session, model_version="v1", threshold=1.5)
    assert result.total_return == pytest.approx(0.0, abs=1e-9)
    assert result.turnover == pytest.approx(0.0, abs=1e-9)


def test_backtest_raises_when_no_predictions(session):
    with pytest.raises(RuntimeError, match="No predictions found"):
        run_backtest(session, model_version="does-not-exist")
