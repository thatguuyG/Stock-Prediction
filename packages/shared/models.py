"""ORM models for the Phase 1 data layer."""
from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Ticker(Base):
    __tablename__ = "tickers"

    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    bars: Mapped[list["PriceBar"]] = relationship(back_populates="ticker")
    indicators: Mapped[list["Indicator"]] = relationship(back_populates="ticker")


class PriceBar(Base):
    __tablename__ = "price_bars"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), ForeignKey("tickers.symbol"), nullable=False)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(32), default="yfinance", nullable=False)

    ticker: Mapped[Ticker] = relationship(back_populates="bars")

    __table_args__ = (
        UniqueConstraint("symbol", "ts", name="uq_price_bars_symbol_ts"),
        Index("ix_price_bars_symbol_ts_desc", "symbol", "ts"),
    )


class Indicator(Base):
    __tablename__ = "indicators"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), ForeignKey("tickers.symbol"), nullable=False)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)

    ticker: Mapped[Ticker] = relationship(back_populates="indicators")

    __table_args__ = (
        UniqueConstraint("symbol", "ts", "name", name="uq_indicators_symbol_ts_name"),
        Index("ix_indicators_symbol_ts_desc", "symbol", "ts"),
    )


class NewsItem(Base):
    __tablename__ = "news_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str | None] = mapped_column(
        String(16), ForeignKey("tickers.symbol"), nullable=True
    )
    source: Mapped[str | None] = mapped_column(String(128), nullable=True)
    headline: Mapped[str] = mapped_column(String(1024), nullable=False)
    url: Mapped[str] = mapped_column(String(2048), nullable=False, unique=True)
    published_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    fetched_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    sentiment: Mapped["Sentiment | None"] = relationship(back_populates="news_item", uselist=False)

    __table_args__ = (
        Index("ix_news_items_symbol_published", "symbol", "published_at"),
    )


class Sentiment(Base):
    __tablename__ = "sentiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    news_item_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("news_items.id"), nullable=False, unique=True
    )
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    compound: Mapped[float] = mapped_column(Float, nullable=False)
    pos: Mapped[float] = mapped_column(Float, nullable=False)
    neu: Mapped[float] = mapped_column(Float, nullable=False)
    neg: Mapped[float] = mapped_column(Float, nullable=False)

    news_item: Mapped[NewsItem] = relationship(back_populates="sentiment")
