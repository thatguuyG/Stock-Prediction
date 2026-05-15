"""signals + orders + trades + positions + risk_state tables for Phase 3 execution

Revision ID: 0003_execution_tables
Revises: 0002_model_tables
Create Date: 2026-05-15
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_execution_tables"
down_revision = "0002_model_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "signals",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "symbol", sa.String(length=16), sa.ForeignKey("tickers.symbol"), nullable=False
        ),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("decision", sa.String(length=8), nullable=False),
        sa.Column("rationale", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "symbol", "ts", "model_version", name="uq_signals_symbol_ts_model"
        ),
    )
    op.create_index("ix_signals_symbol_ts", "signals", ["symbol", "ts"])
    op.create_index("ix_signals_model_version", "signals", ["model_version"])
    op.create_index("ix_signals_decision", "signals", ["decision"])

    op.create_table(
        "orders",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "symbol", sa.String(length=16), sa.ForeignKey("tickers.symbol"), nullable=False
        ),
        sa.Column("side", sa.String(length=8), nullable=False),
        sa.Column("qty", sa.Float(), nullable=False),
        sa.Column(
            "order_type", sa.String(length=16), nullable=False, server_default="market"
        ),
        sa.Column("limit_price", sa.Float(), nullable=True),
        sa.Column("stop_price", sa.Float(), nullable=True),
        sa.Column("take_profit", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="pending"),
        sa.Column(
            "submitted_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("filled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("broker_order_id", sa.String(length=64), nullable=True, unique=True),
        sa.Column("signal_id", sa.Integer(), sa.ForeignKey("signals.id"), nullable=True),
    )
    op.create_index("ix_orders_status", "orders", ["status"])
    op.create_index("ix_orders_symbol_submitted_at", "orders", ["symbol", "submitted_at"])

    op.create_table(
        "trades",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("order_id", sa.Integer(), sa.ForeignKey("orders.id"), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("qty", sa.Float(), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("fee", sa.Float(), nullable=False, server_default="0"),
        sa.Column("broker_trade_id", sa.String(length=64), nullable=True, unique=True),
    )
    op.create_index("ix_trades_order_id", "trades", ["order_id"])

    op.create_table(
        "positions",
        sa.Column(
            "symbol", sa.String(length=16), sa.ForeignKey("tickers.symbol"), primary_key=True
        ),
        sa.Column("qty", sa.Float(), nullable=False, server_default="0"),
        sa.Column("avg_price", sa.Float(), nullable=False, server_default="0"),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("source", sa.String(length=16), nullable=False, server_default="alpaca"),
    )

    op.create_table(
        "risk_state",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "ts",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("equity", sa.Float(), nullable=False),
        sa.Column("cash", sa.Float(), nullable=False),
        sa.Column("exposure_pct", sa.Float(), nullable=False),
        sa.Column("max_drawdown", sa.Float(), nullable=False, server_default="0"),
        sa.Column("n_open_positions", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("halted", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("notes", sa.Text(), nullable=True),
    )
    op.create_index("ix_risk_state_ts", "risk_state", ["ts"])


def downgrade() -> None:
    op.drop_index("ix_risk_state_ts", table_name="risk_state")
    op.drop_table("risk_state")
    op.drop_table("positions")
    op.drop_index("ix_trades_order_id", table_name="trades")
    op.drop_table("trades")
    op.drop_index("ix_orders_symbol_submitted_at", table_name="orders")
    op.drop_index("ix_orders_status", table_name="orders")
    op.drop_table("orders")
    op.drop_index("ix_signals_decision", table_name="signals")
    op.drop_index("ix_signals_model_version", table_name="signals")
    op.drop_index("ix_signals_symbol_ts", table_name="signals")
    op.drop_table("signals")
