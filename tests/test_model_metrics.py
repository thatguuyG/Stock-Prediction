"""Hand-rolled metrics: assert against hand-computed values."""
from __future__ import annotations

import math

import numpy as np
import pytest

from services.model import metrics as M


def test_sharpe_known_input():
    # mean=0.001, std=0.01 → daily sharpe 0.1; annualized = 0.1*sqrt(252) ≈ 1.587
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.01, 100_000)
    sharpe = M.sharpe(returns)
    assert sharpe == pytest.approx(0.1 * math.sqrt(252), rel=0.05)


def test_sharpe_zero_std_returns_zero():
    assert M.sharpe(np.array([0.01, 0.01, 0.01])) == 0.0


def test_sharpe_empty_returns_zero():
    assert M.sharpe(np.array([])) == 0.0


def test_max_drawdown_known_input():
    eq = np.array([1.0, 1.1, 1.2, 0.9, 1.0, 1.3])
    # peak at 1.2, trough at 0.9 → dd = 0.3/1.2 = 0.25
    assert M.max_drawdown(eq) == pytest.approx(0.25)


def test_max_drawdown_monotonic_zero():
    assert M.max_drawdown(np.array([1.0, 1.1, 1.2, 1.5])) == 0.0


def test_hit_rate_simple():
    assert M.hit_rate(np.array([0.01, -0.005, 0.0, 0.02, -0.01])) == pytest.approx(0.4)


def test_total_return_known():
    r = np.array([0.10, -0.05, 0.20])  # 1.1 * 0.95 * 1.2 - 1 = 0.254
    assert M.total_return(r) == pytest.approx(0.254, abs=1e-9)


def test_log_loss_perfect_predictions():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.01, 0.99, 0.01, 0.99])
    assert M.log_loss(y_true, y_prob) < 0.05


def test_log_loss_random():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5])
    # -log(0.5) ≈ 0.693
    assert M.log_loss(y_true, y_prob) == pytest.approx(math.log(2), abs=1e-9)


def test_accuracy_simple():
    assert M.accuracy(np.array([1, 0, 1, 1]), np.array([1, 0, 0, 1])) == 0.75
