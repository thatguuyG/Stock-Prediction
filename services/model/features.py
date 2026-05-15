"""Feature matrix construction for Phase 2 model training.

Joins price_bars + indicators (long → wide) + sentiments into a single
DataFrame keyed on (symbol, ts), suitable for sklearn-style training.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from packages.shared.logging import get_logger
from packages.shared.models import Indicator, NewsItem, PriceBar, Sentiment

log = get_logger(__name__)

INDICATOR_NAMES = [
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
]

RETURN_LAGS = [1, 5, 20]
SENTIMENT_WINDOWS = [1, 3, 7]
VOL_ZSCORE_WINDOW = 20

FEATURE_COLUMNS: list[str] = (
    INDICATOR_NAMES
    + [f"ret_{lag}d" for lag in RETURN_LAGS]
    + [f"sent_mean_{w}d" for w in SENTIMENT_WINDOWS]
    + [f"vol_zscore_{VOL_ZSCORE_WINDOW}d"]
)


def _load_prices(session: Session, since: dt.date | None) -> pd.DataFrame:
    stmt = select(
        PriceBar.symbol, PriceBar.ts, PriceBar.close, PriceBar.volume
    ).order_by(PriceBar.symbol, PriceBar.ts)
    if since is not None:
        stmt = stmt.where(PriceBar.ts >= dt.datetime.combine(since, dt.time.min))
    rows = session.execute(stmt).all()
    if not rows:
        return pd.DataFrame(columns=["symbol", "ts", "close", "volume"])
    return pd.DataFrame(rows, columns=["symbol", "ts", "close", "volume"])


def _load_indicators_wide(session: Session, since: dt.date | None) -> pd.DataFrame:
    stmt = select(
        Indicator.symbol, Indicator.ts, Indicator.name, Indicator.value
    ).where(Indicator.name.in_(INDICATOR_NAMES))
    if since is not None:
        stmt = stmt.where(Indicator.ts >= dt.datetime.combine(since, dt.time.min))
    rows = session.execute(stmt).all()
    if not rows:
        return pd.DataFrame(columns=["symbol", "ts"] + INDICATOR_NAMES)
    long_df = pd.DataFrame(rows, columns=["symbol", "ts", "name", "value"])
    wide = long_df.pivot_table(
        index=["symbol", "ts"], columns="name", values="value", aggfunc="first"
    )
    for name in INDICATOR_NAMES:
        if name not in wide.columns:
            wide[name] = np.nan
    return wide[INDICATOR_NAMES].reset_index()


def _load_daily_sentiment(session: Session, since: dt.date | None) -> pd.DataFrame:
    """Return one row per (symbol, calendar_day) with mean compound score."""
    stmt = select(
        NewsItem.symbol, NewsItem.published_at, Sentiment.compound
    ).join(Sentiment, Sentiment.news_item_id == NewsItem.id)
    if since is not None:
        stmt = stmt.where(NewsItem.published_at >= dt.datetime.combine(since, dt.time.min))
    rows = session.execute(stmt).all()
    if not rows:
        return pd.DataFrame(columns=["symbol", "day", "compound"])
    df = pd.DataFrame(rows, columns=["symbol", "published_at", "compound"])
    df = df.dropna(subset=["symbol", "published_at"])
    if df.empty:
        return pd.DataFrame(columns=["symbol", "day", "compound"])
    df["day"] = pd.to_datetime(df["published_at"]).dt.tz_localize(None).dt.normalize()
    return df.groupby(["symbol", "day"], as_index=False)["compound"].mean()


def _merge_sentiment(prices: pd.DataFrame, sent_daily: pd.DataFrame) -> pd.DataFrame:
    """Attach sent_mean_{1,3,7}d rolling means to each (symbol, ts).

    Missing news on a trading day is treated as compound=0 (neutral).
    """
    out = prices.copy()
    out["_day"] = pd.to_datetime(out["ts"]).dt.tz_localize(None).dt.normalize()
    if sent_daily.empty:
        for window in SENTIMENT_WINDOWS:
            out[f"sent_mean_{window}d"] = 0.0
        return out.drop(columns=["_day"])

    merged = out.merge(
        sent_daily.rename(columns={"day": "_day"}),
        on=["symbol", "_day"],
        how="left",
    )
    merged["compound"] = merged["compound"].fillna(0.0)
    for window in SENTIMENT_WINDOWS:
        merged[f"sent_mean_{window}d"] = (
            merged.groupby("symbol", group_keys=False)["compound"]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
    return merged.drop(columns=["_day", "compound"])


def _add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lagged log returns + 20-day volume z-score + target label."""
    out = df.copy().sort_values(["symbol", "ts"]).reset_index(drop=True)
    g = out.groupby("symbol", group_keys=False)

    log_close = np.log(out["close"])
    for lag in RETURN_LAGS:
        out[f"ret_{lag}d"] = (
            log_close.groupby(out["symbol"]).diff(lag)
        )

    vol_mean = g["volume"].transform(
        lambda s: s.rolling(VOL_ZSCORE_WINDOW, min_periods=VOL_ZSCORE_WINDOW).mean()
    )
    vol_std = g["volume"].transform(
        lambda s: s.rolling(VOL_ZSCORE_WINDOW, min_periods=VOL_ZSCORE_WINDOW).std()
    )
    out[f"vol_zscore_{VOL_ZSCORE_WINDOW}d"] = (out["volume"] - vol_mean) / vol_std

    next_close = g["close"].shift(-1)
    out["target"] = (next_close > out["close"]).astype("Int64")
    out.loc[next_close.isna(), "target"] = pd.NA
    return out


def build_feature_matrix(
    session: Session, since: dt.date | None = None
) -> pd.DataFrame:
    """Construct the (symbol, ts, features..., target) matrix.

    Drops rows with any NaN in feature columns or a missing target (last row
    per symbol). Caller is responsible for downstream train/test splitting.
    """
    prices = _load_prices(session, since)
    if prices.empty:
        log.warning("build_feature_matrix: no price bars found")
        return pd.DataFrame(columns=["symbol", "ts", "target"] + FEATURE_COLUMNS)

    indicators_wide = _load_indicators_wide(session, since)
    sent_daily = _load_daily_sentiment(session, since)

    base = _add_price_features(prices)
    merged = base.merge(indicators_wide, on=["symbol", "ts"], how="left")
    merged = _merge_sentiment(merged, sent_daily)

    cols = ["symbol", "ts", "target"] + FEATURE_COLUMNS
    out = merged[cols]
    out = out.dropna(subset=FEATURE_COLUMNS + ["target"]).reset_index(drop=True)
    out["target"] = out["target"].astype(int)
    return out
