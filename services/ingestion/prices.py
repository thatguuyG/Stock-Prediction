"""OHLCV price ingestion via yfinance."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import yfinance as yf
from sqlalchemy import select
from sqlalchemy.orm import Session

from packages.shared.db import upsert_ignore
from packages.shared.logging import get_logger
from packages.shared.models import PriceBar, Ticker

log = get_logger(__name__)


def fetch_ohlcv(symbol: str, start: dt.date, end: dt.date | None = None) -> pd.DataFrame:
    """Pull daily OHLCV bars from yfinance. Returns DataFrame indexed by tz-aware UTC datetime."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(
        start=start.isoformat(),
        end=end.isoformat() if end else None,
        interval="1d",
        auto_adjust=False,
    )
    if df.empty:
        log.warning("yfinance returned no rows for %s", symbol)
        return df
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df = df[["open", "high", "low", "close", "volume"]].copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def ensure_ticker(session: Session, symbol: str) -> None:
    existing = session.execute(select(Ticker).where(Ticker.symbol == symbol)).scalar_one_or_none()
    if existing is None:
        session.add(Ticker(symbol=symbol, active=True))
        session.flush()


def upsert_bars(session: Session, symbol: str, df: pd.DataFrame) -> int:
    """Insert bars idempotently. Returns count of newly-inserted rows."""
    if df.empty:
        return 0
    ensure_ticker(session, symbol)
    rows = [
        {
            "symbol": symbol,
            "ts": ts.to_pydatetime(),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "source": "yfinance",
        }
        for ts, row in df.iterrows()
    ]
    return upsert_ignore(session, PriceBar.__table__, rows, ["symbol", "ts"])


def ingest_symbol(session: Session, symbol: str, since: dt.date) -> int:
    df = fetch_ohlcv(symbol, since)
    inserted = upsert_bars(session, symbol, df)
    log.info("Ingested %d new bars for %s (since %s)", inserted, symbol, since)
    return inserted
