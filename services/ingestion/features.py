"""Technical indicator computation via pandas-ta."""
from __future__ import annotations

import pandas as pd
import pandas_ta as ta
from sqlalchemy import select
from sqlalchemy.orm import Session

from packages.shared.db import upsert_ignore
from packages.shared.logging import get_logger
from packages.shared.models import Indicator, PriceBar

log = get_logger(__name__)

INDICATOR_COLUMNS = {
    "RSI_14": "rsi_14",
    "MACD_12_26_9": "macd",
    "MACDh_12_26_9": "macd_hist",
    "MACDs_12_26_9": "macd_signal",
    # pandas-ta's bbands column naming has varied across versions; accept both.
    "BBL_20_2.0": "bb_lower",
    "BBM_20_2.0": "bb_mid",
    "BBU_20_2.0": "bb_upper",
    "BBL_20_2.0_2.0": "bb_lower",
    "BBM_20_2.0_2.0": "bb_mid",
    "BBU_20_2.0_2.0": "bb_upper",
    "SMA_20": "sma_20",
    "SMA_50": "sma_50",
    "SMA_200": "sma_200",
    "EMA_12": "ema_12",
    "EMA_26": "ema_26",
}


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all Phase 1 indicators. Input must have columns: open, high, low, close, volume."""
    if df.empty:
        return df
    out = df.copy()
    out.ta.rsi(length=14, append=True)
    out.ta.macd(fast=12, slow=26, signal=9, append=True)
    out.ta.bbands(length=20, std=2, append=True)
    for length in (20, 50, 200):
        out.ta.sma(length=length, append=True)
    for length in (12, 26):
        out.ta.ema(length=length, append=True)
    keep = [c for c in INDICATOR_COLUMNS if c in out.columns]
    return out[keep].rename(columns=INDICATOR_COLUMNS)


def load_bars(session: Session, symbol: str) -> pd.DataFrame:
    rows = session.execute(
        select(PriceBar.ts, PriceBar.open, PriceBar.high, PriceBar.low, PriceBar.close, PriceBar.volume)
        .where(PriceBar.symbol == symbol)
        .order_by(PriceBar.ts.asc())
    ).all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.set_index("ts")
    return df


def upsert_indicators(session: Session, symbol: str, indicators_df: pd.DataFrame) -> int:
    """Write long-form indicator rows, skipping NaN values. Returns count inserted."""
    if indicators_df.empty:
        return 0
    rows: list[dict] = []
    for ts, row in indicators_df.iterrows():
        ts_py = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        for name, value in row.items():
            if pd.isna(value):
                continue
            rows.append(
                {"symbol": symbol, "ts": ts_py, "name": name, "value": float(value)}
            )
    return upsert_ignore(session, Indicator.__table__, rows, ["symbol", "ts", "name"])


def compute_for_symbol(session: Session, symbol: str) -> int:
    bars = load_bars(session, symbol)
    if bars.empty:
        log.warning("No bars found for %s — skipping indicator computation", symbol)
        return 0
    ind = compute_indicators(bars)
    inserted = upsert_indicators(session, symbol, ind)
    log.info("Wrote %d indicator rows for %s", inserted, symbol)
    return inserted
