"""initial schema: tickers, price_bars, indicators, news_items, sentiments

Revision ID: 0001_initial
Revises:
Create Date: 2026-05-15
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tickers",
        sa.Column("symbol", sa.String(length=16), primary_key=True),
        sa.Column("name", sa.String(length=256), nullable=True),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "price_bars",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "symbol", sa.String(length=16), sa.ForeignKey("tickers.symbol"), nullable=False
        ),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Float(), nullable=False),
        sa.Column("high", sa.Float(), nullable=False),
        sa.Column("low", sa.Float(), nullable=False),
        sa.Column("close", sa.Float(), nullable=False),
        sa.Column("volume", sa.Float(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False, server_default="yfinance"),
        sa.UniqueConstraint("symbol", "ts", name="uq_price_bars_symbol_ts"),
    )
    op.create_index("ix_price_bars_symbol_ts_desc", "price_bars", ["symbol", "ts"])

    op.create_table(
        "indicators",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "symbol", sa.String(length=16), sa.ForeignKey("tickers.symbol"), nullable=False
        ),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.UniqueConstraint("symbol", "ts", "name", name="uq_indicators_symbol_ts_name"),
    )
    op.create_index("ix_indicators_symbol_ts_desc", "indicators", ["symbol", "ts"])

    op.create_table(
        "news_items",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "symbol", sa.String(length=16), sa.ForeignKey("tickers.symbol"), nullable=True
        ),
        sa.Column("source", sa.String(length=128), nullable=True),
        sa.Column("headline", sa.String(length=1024), nullable=False),
        sa.Column("url", sa.String(length=2048), nullable=False, unique=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_news_items_symbol_published", "news_items", ["symbol", "published_at"])

    op.create_table(
        "sentiments",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "news_item_id",
            sa.Integer(),
            sa.ForeignKey("news_items.id"),
            nullable=False,
            unique=True,
        ),
        sa.Column("model", sa.String(length=64), nullable=False),
        sa.Column("compound", sa.Float(), nullable=False),
        sa.Column("pos", sa.Float(), nullable=False),
        sa.Column("neu", sa.Float(), nullable=False),
        sa.Column("neg", sa.Float(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("sentiments")
    op.drop_index("ix_news_items_symbol_published", table_name="news_items")
    op.drop_table("news_items")
    op.drop_index("ix_indicators_symbol_ts_desc", table_name="indicators")
    op.drop_table("indicators")
    op.drop_index("ix_price_bars_symbol_ts_desc", table_name="price_bars")
    op.drop_table("price_bars")
    op.drop_table("tickers")
