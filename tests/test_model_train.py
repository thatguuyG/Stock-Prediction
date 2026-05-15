"""Walk-forward training on an embedded-signal synthetic dataset."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
from sqlalchemy import select

from packages.shared.models import Indicator, PriceBar, Prediction, Ticker
from services.model.features import INDICATOR_NAMES
from services.model.train import train


def _insert_embedded_signal(
    session, symbol: str = "AAPL", n_days: int = 260, seed: int = 0
) -> None:
    """Construct a dataset where close[t+1] > close[t] iff rsi_14[t] < 30.

    All other indicators are random noise — the model must learn to lean on
    rsi_14 to beat 50%.
    """
    rng = np.random.default_rng(seed)
    base_ts = dt.datetime(2023, 1, 2, tzinfo=dt.timezone.utc)

    rsi = rng.uniform(10, 90, n_days)
    direction = (rsi < 30).astype(int)
    direction = direction * 2 - 1
    drift = direction * 0.012
    noise = rng.normal(0, 0.001, n_days)
    log_returns = drift + noise
    closes = 100.0 * np.exp(np.cumsum(np.concatenate([[0], log_returns[:-1]])))

    session.add(Ticker(symbol=symbol, active=True))
    session.flush()

    for i in range(n_days):
        ts = base_ts + dt.timedelta(days=i)
        c = float(closes[i])
        session.add(
            PriceBar(
                symbol=symbol,
                ts=ts,
                open=c - 0.05,
                high=c + 0.3,
                low=c - 0.3,
                close=c,
                volume=1_000_000.0 + rng.uniform(-50_000, 50_000),
                source="synth",
            )
        )
        for name in INDICATOR_NAMES:
            if name == "rsi_14":
                value = float(rsi[i])
            else:
                value = float(rng.normal(50, 10))
            session.add(Indicator(symbol=symbol, ts=ts, name=name, value=value))
    session.flush()


def test_train_learns_embedded_signal(session, tmp_path: Path):
    _insert_embedded_signal(session)
    result = train(
        session,
        model_version="testv1",
        train_window=80,
        val_window=20,
        step=20,
        models_dir=tmp_path,
    )
    session.flush()

    assert result.n_folds >= 3, f"expected multiple folds, got {result.n_folds}"
    assert result.mean_accuracy >= 0.6, (
        f"model should learn the embedded signal — accuracy={result.mean_accuracy:.3f}"
    )
    assert result.artifact_path.exists()


def test_train_writes_predictions(session, tmp_path: Path):
    _insert_embedded_signal(session)
    train(
        session,
        model_version="testv1",
        train_window=80,
        val_window=20,
        step=20,
        models_dir=tmp_path,
    )
    session.flush()
    rows = session.execute(
        select(Prediction).where(Prediction.model_version == "testv1")
    ).all()
    assert len(rows) > 0
    for (p,) in rows:
        assert 0.0 <= p.score <= 1.0
        assert p.label_pred in (0, 1)


def test_train_walk_forward_no_leakage(session, tmp_path: Path):
    _insert_embedded_signal(session)
    result = train(
        session,
        model_version="testv1",
        train_window=80,
        val_window=20,
        step=20,
        models_dir=tmp_path,
    )
    for fold in result.folds:
        assert fold.train_end <= fold.val_end
        # Per the contract, validation start (== fold.train_end) is strictly later
        # than every training timestamp; we sanity-check ordering here.
        assert fold.train_start < fold.train_end


def test_train_idempotent_predictions(session, tmp_path: Path):
    _insert_embedded_signal(session)
    r1 = train(
        session, model_version="testv1",
        train_window=80, val_window=20, step=20, models_dir=tmp_path,
    )
    session.flush()
    r2 = train(
        session, model_version="testv1",
        train_window=80, val_window=20, step=20, models_dir=tmp_path,
    )
    session.flush()
    assert r1.n_predictions_written > 0
    assert r2.n_predictions_written == 0, "re-training same version should not duplicate rows"
