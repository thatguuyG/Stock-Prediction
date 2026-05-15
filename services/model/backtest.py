"""Pure-pandas walk-forward backtester.

Reads `predictions` rows for a given model_version, joins to price_bars,
computes daily portfolio returns under an equal-weight / position-capped
long-only policy, and writes a `backtest_runs` row.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from packages.shared.logging import get_logger
from packages.shared.models import BacktestRun, Prediction, PriceBar
from services.model import metrics as M

log = get_logger(__name__)

DEFAULT_THRESHOLD = 0.55
DEFAULT_SLIPPAGE_BPS = 5
DEFAULT_MAX_WEIGHT = 0.20
PERIODS_PER_YEAR = 252


@dataclass
class BacktestResult:
    model_version: str
    threshold: float
    sharpe: float
    max_drawdown: float
    hit_rate: float
    total_return: float
    turnover: float
    n_trades: int
    n_days: int
    n_symbols: int


def _load_joined(session: Session, model_version: str) -> pd.DataFrame:
    rows = session.execute(
        select(
            Prediction.symbol,
            Prediction.ts,
            Prediction.score,
            PriceBar.close,
        )
        .join(
            PriceBar,
            (PriceBar.symbol == Prediction.symbol) & (PriceBar.ts == Prediction.ts),
        )
        .where(Prediction.model_version == model_version)
        .order_by(Prediction.symbol, Prediction.ts)
    ).all()
    if not rows:
        return pd.DataFrame(columns=["symbol", "ts", "score", "close"])
    return pd.DataFrame(rows, columns=["symbol", "ts", "score", "close"])


def _compute_weights(
    panel: pd.DataFrame, threshold: float, max_weight: float
) -> pd.DataFrame:
    """Wide weights table: rows = ts, cols = symbol."""
    panel = panel.copy()
    panel["signal"] = (panel["score"] > threshold).astype(float)

    by_ts = panel.groupby("ts")["signal"].transform("sum")
    raw_weight = panel["signal"] / by_ts.replace(0, np.nan)
    panel["weight"] = np.minimum(raw_weight.fillna(0.0), max_weight)

    weights = panel.pivot_table(
        index="ts", columns="symbol", values="weight", aggfunc="first"
    ).fillna(0.0)
    return weights.sort_index()


def _compute_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Wide next-day returns: returns.loc[t, sym] = close[t+1] / close[t] - 1."""
    closes = panel.pivot_table(
        index="ts", columns="symbol", values="close", aggfunc="first"
    ).sort_index()
    next_close = closes.shift(-1)
    return (next_close / closes - 1.0).fillna(0.0)


def _run_pipeline(
    panel: pd.DataFrame,
    threshold: float,
    slippage_bps: float,
    max_weight: float,
) -> tuple[pd.Series, pd.Series, float, int]:
    """Return (portfolio_returns, equity_curve, turnover, n_trades)."""
    weights = _compute_weights(panel, threshold, max_weight)
    returns_wide = _compute_returns(panel)

    aligned_returns = returns_wide.reindex_like(weights).fillna(0.0)
    gross_daily = (weights * aligned_returns).sum(axis=1)

    weight_changes = weights.diff().abs().fillna(weights.abs())
    slippage_cost = weight_changes.sum(axis=1) * (slippage_bps / 10_000.0)
    net_daily = gross_daily - slippage_cost

    equity_curve = (1.0 + net_daily).cumprod()
    turnover = float(weight_changes.values.sum())
    n_trades = int((weight_changes > 0).values.sum())
    return net_daily, equity_curve, turnover, n_trades


def run_backtest(
    session: Session,
    model_version: str,
    threshold: float = DEFAULT_THRESHOLD,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    notes: str | None = None,
) -> BacktestResult:
    panel = _load_joined(session, model_version)
    if panel.empty:
        raise RuntimeError(
            f"No predictions found for model_version={model_version!r}. "
            "Train or predict before backtesting."
        )

    started_at = dt.datetime.now(dt.timezone.utc)
    net_daily, equity_curve, turnover, n_trades = _run_pipeline(
        panel, threshold, slippage_bps, max_weight
    )

    result = BacktestResult(
        model_version=model_version,
        threshold=threshold,
        sharpe=M.sharpe(net_daily.values, PERIODS_PER_YEAR),
        max_drawdown=M.max_drawdown(equity_curve.values),
        hit_rate=M.hit_rate(net_daily.values),
        total_return=M.total_return(net_daily.values),
        turnover=turnover,
        n_trades=n_trades,
        n_days=len(net_daily),
        n_symbols=int(panel["symbol"].nunique()),
    )

    session.add(
        BacktestRun(
            model_version=model_version,
            started_at=started_at,
            finished_at=dt.datetime.now(dt.timezone.utc),
            threshold=threshold,
            sharpe=result.sharpe,
            max_drawdown=result.max_drawdown,
            hit_rate=result.hit_rate,
            total_return=result.total_return,
            turnover=turnover,
            n_trades=n_trades,
            notes=notes,
        )
    )
    session.flush()
    log.info(
        "Backtest %s: sharpe=%.3f total_return=%.3f max_dd=%.3f hit_rate=%.3f n_trades=%d",
        model_version,
        result.sharpe,
        result.total_return,
        result.max_drawdown,
        result.hit_rate,
        n_trades,
    )
    return result
