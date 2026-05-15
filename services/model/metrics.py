"""Backtest and classification metrics.

Hand-rolled to avoid a sklearn dependency. All functions take numpy arrays or
pandas Series and return plain floats.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe(returns: pd.Series | np.ndarray, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio of a return series (risk-free rate assumed 0)."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    std = r.std(ddof=0)
    if std == 0:
        return 0.0
    return float(r.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series | np.ndarray) -> float:
    """Maximum peak-to-trough decline as a positive fraction (0.20 = 20% drawdown)."""
    eq = np.asarray(equity_curve, dtype=float)
    if eq.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(eq)
    drawdowns = (running_peak - eq) / running_peak
    return float(drawdowns.max())


def hit_rate(returns: pd.Series | np.ndarray) -> float:
    """Fraction of return observations strictly greater than zero."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    return float((r > 0).mean())


def total_return(returns: pd.Series | np.ndarray) -> float:
    """Compounded total return over a daily return series."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    return float(np.prod(1.0 + r) - 1.0)


def log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """Binary cross-entropy. y_true in {0, 1}, y_prob in [0, 1]."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    return float(-np.mean(yt * np.log(yp) + (1 - yt) * np.log(1 - yp)))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.size == 0:
        return 0.0
    return float((yt == yp).mean())
