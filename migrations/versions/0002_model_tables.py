"""predictions + backtest_runs tables for Phase 2 model layer

Revision ID: 0002_model_tables
Revises: 0001_initial
Create Date: 2026-05-15
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_model_tables"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "symbol", sa.String(length=16), sa.ForeignKey("tickers.symbol"), nullable=False
        ),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("label_pred", sa.Integer(), nullable=False),
        sa.UniqueConstraint(
            "symbol", "ts", "model_version", name="uq_predictions_symbol_ts_model"
        ),
    )
    op.create_index("ix_predictions_symbol_ts", "predictions", ["symbol", "ts"])
    op.create_index("ix_predictions_model_version", "predictions", ["model_version"])

    op.create_table(
        "backtest_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("sharpe", sa.Float(), nullable=False),
        sa.Column("max_drawdown", sa.Float(), nullable=False),
        sa.Column("hit_rate", sa.Float(), nullable=False),
        sa.Column("total_return", sa.Float(), nullable=False),
        sa.Column("turnover", sa.Float(), nullable=False),
        sa.Column("n_trades", sa.Integer(), nullable=False),
        sa.Column("notes", sa.String(length=1024), nullable=True),
    )
    op.create_index("ix_backtest_runs_model_version", "backtest_runs", ["model_version"])


def downgrade() -> None:
    op.drop_index("ix_backtest_runs_model_version", table_name="backtest_runs")
    op.drop_table("backtest_runs")
    op.drop_index("ix_predictions_model_version", table_name="predictions")
    op.drop_index("ix_predictions_symbol_ts", table_name="predictions")
    op.drop_table("predictions")
