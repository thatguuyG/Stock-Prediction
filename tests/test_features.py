"""Indicator engine tests — deterministic price series, sanity bounds on indicators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from packages.shared.models import Indicator, Ticker
from services.ingestion import features


@pytest.fixture(name="long_bars")
def fixture_long_bars() -> pd.DataFrame:
    n = 250
    idx = pd.date_range("2023-01-02", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0.1, 1.0, n))
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = close + rng.normal(0, 0.3, n)
    volume = rng.uniform(1e6, 5e6, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_compute_indicators_produces_expected_columns(long_bars):
    ind = features.compute_indicators(long_bars)
    expected = {
        "rsi_14",
        "macd",
        "macd_hist",
        "macd_signal",
        "bb_lower",
        "bb_mid",
        "bb_upper",
        "sma_20",
        "sma_50",
        "sma_200",
        "ema_12",
        "ema_26",
    }
    assert expected.issubset(set(ind.columns))


def test_rsi_in_valid_range(long_bars):
    ind = features.compute_indicators(long_bars)
    rsi = ind["rsi_14"].dropna()
    assert len(rsi) > 0
    assert ((rsi >= 0) & (rsi <= 100)).all()


def test_bollinger_ordering(long_bars):
    ind = features.compute_indicators(long_bars).dropna()
    assert (ind["bb_lower"] <= ind["bb_mid"]).all()
    assert (ind["bb_mid"] <= ind["bb_upper"]).all()


def test_upsert_indicators_idempotent_and_skips_nan(session, long_bars):
    session.add(Ticker(symbol="AAPL", active=True))
    session.flush()

    ind = features.compute_indicators(long_bars)
    n1 = features.upsert_indicators(session, "AAPL", ind)
    session.flush()
    n2 = features.upsert_indicators(session, "AAPL", ind)
    session.flush()

    assert n1 > 0
    assert n2 == 0

    rows = session.execute(select(Indicator).where(Indicator.symbol == "AAPL")).all()
    values = [r[0].value for r in rows]
    assert all(not np.isnan(v) for v in values)
