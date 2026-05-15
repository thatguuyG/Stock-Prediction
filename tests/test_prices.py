"""Price ingestion tests — yfinance mocked, idempotent upsert verified."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
from sqlalchemy import select

from packages.shared.models import PriceBar, Ticker
from services.ingestion import prices


@pytest.fixture(name="fake_ohlcv")
def fixture_fake_ohlcv() -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=5, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "Open": [100.0, 101, 102, 103, 104],
            "High": [101.0, 102, 103, 104, 105],
            "Low": [99.0, 100, 101, 102, 103],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "Volume": [1000.0, 1100, 1200, 1300, 1400],
        },
        index=idx,
    )


def test_fetch_ohlcv_normalizes_columns(monkeypatch, fake_ohlcv):
    class FakeTicker:
        def __init__(self, _):
            pass

        def history(self, **_kwargs):
            return fake_ohlcv

    monkeypatch.setattr(prices.yf, "Ticker", FakeTicker)

    df = prices.fetch_ohlcv("AAPL", dt.date(2024, 1, 1))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 5
    assert str(df.index.tz) == "UTC"


def test_upsert_bars_is_idempotent(session, fake_ohlcv):
    df = fake_ohlcv.rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    )[["open", "high", "low", "close", "volume"]]

    inserted_first = prices.upsert_bars(session, "AAPL", df)
    session.flush()
    inserted_second = prices.upsert_bars(session, "AAPL", df)
    session.flush()

    assert inserted_first == 5
    assert inserted_second == 0

    total = session.execute(select(PriceBar).where(PriceBar.symbol == "AAPL")).all()
    assert len(total) == 5

    ticker = session.execute(select(Ticker).where(Ticker.symbol == "AAPL")).scalar_one()
    assert ticker.active is True


def test_upsert_bars_empty_df_noop(session):
    n = prices.upsert_bars(session, "AAPL", pd.DataFrame())
    assert n == 0
