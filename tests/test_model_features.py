"""Feature matrix construction tests — synthetic data in SQLite session."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from packages.shared.models import Indicator, NewsItem, PriceBar, Sentiment, Ticker
from services.model.features import (
    FEATURE_COLUMNS,
    INDICATOR_NAMES,
    build_feature_matrix,
)


def _insert_synth(
    session,
    symbol: str,
    n_days: int = 260,
    seed: int = 0,
    with_news: bool = True,
):
    rng = np.random.default_rng(seed)
    base_ts = dt.datetime(2024, 1, 2, tzinfo=dt.timezone.utc)
    closes = 100 + np.cumsum(rng.normal(0.05, 1.0, n_days))
    session.add(Ticker(symbol=symbol, active=True))
    session.flush()

    for i in range(n_days):
        ts = base_ts + dt.timedelta(days=i)
        c = float(closes[i])
        session.add(
            PriceBar(
                symbol=symbol,
                ts=ts,
                open=c - 0.1,
                high=c + 0.5,
                low=c - 0.5,
                close=c,
                volume=1_000_000.0 + rng.uniform(-100_000, 100_000),
                source="synth",
            )
        )
        for name in INDICATOR_NAMES:
            session.add(
                Indicator(symbol=symbol, ts=ts, name=name, value=float(rng.normal(50, 5)))
            )

    if with_news:
        for i in range(0, n_days, 7):
            ts = base_ts + dt.timedelta(days=i)
            ni = NewsItem(
                symbol=symbol,
                source="synth-wire",
                headline=f"{symbol} news day {i}",
                url=f"https://e.example/{symbol}/{i}",
                published_at=ts,
            )
            session.add(ni)
            session.flush()
            session.add(
                Sentiment(
                    news_item_id=ni.id,
                    model="vader-1.0",
                    compound=float(rng.uniform(-0.5, 0.5)),
                    pos=0.3,
                    neu=0.4,
                    neg=0.3,
                )
            )
    session.flush()


def test_feature_matrix_has_expected_columns(session):
    _insert_synth(session, "AAPL")
    df = build_feature_matrix(session)
    assert not df.empty
    assert set(["symbol", "ts", "target"]).issubset(df.columns)
    for col in FEATURE_COLUMNS:
        assert col in df.columns, f"missing feature column: {col}"


def test_feature_matrix_drops_warmup_rows(session):
    _insert_synth(session, "AAPL", n_days=260)
    df = build_feature_matrix(session)
    # 20-day return + 20-day vol z-score warmup + last row drop (no next-day target)
    assert len(df) < 260
    assert len(df) >= 230
    for col in FEATURE_COLUMNS:
        assert df[col].isna().sum() == 0


def test_target_matches_next_day_close_direction(session):
    _insert_synth(session, "AAPL")
    df = build_feature_matrix(session).sort_values("ts").reset_index(drop=True)

    closes = pd.DataFrame(
        session.query(PriceBar.ts, PriceBar.close)  # type: ignore[attr-defined]
        .filter(PriceBar.symbol == "AAPL")
        .order_by(PriceBar.ts)
        .all(),
        columns=["ts", "close"],
    )
    closes["next_close"] = closes["close"].shift(-1)
    closes["expected_target"] = (closes["next_close"] > closes["close"]).astype(int)

    merged = df.merge(closes, on="ts", how="inner")
    assert (merged["target"] == merged["expected_target"]).all()


def test_no_news_yields_zero_sentiment(session):
    _insert_synth(session, "AAPL", with_news=False)
    df = build_feature_matrix(session)
    for col in ["sent_mean_1d", "sent_mean_3d", "sent_mean_7d"]:
        assert (df[col] == 0.0).all(), f"{col} should be 0 with no news"


def test_multi_symbol_features_isolated(session):
    _insert_synth(session, "AAPL", seed=1)
    _insert_synth(session, "MSFT", seed=2)
    df = build_feature_matrix(session)
    syms = set(df["symbol"].unique())
    assert syms == {"AAPL", "MSFT"}
    # last row per symbol should be dropped (no target)
    counts = df.groupby("symbol").size()
    assert counts["AAPL"] == counts["MSFT"]


def test_empty_session_returns_empty(session):
    df = build_feature_matrix(session)
    assert df.empty


def test_since_filter_respected(session):
    _insert_synth(session, "AAPL")
    cutoff = dt.date(2024, 6, 1)
    df = build_feature_matrix(session, since=cutoff)
    assert not df.empty
    cutoff_ts = pd.Timestamp(cutoff)
    ts_naive = pd.to_datetime(df["ts"]).dt.tz_localize(None)
    assert (ts_naive >= cutoff_ts).all()
