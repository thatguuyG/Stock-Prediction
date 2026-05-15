"""Walk-forward XGBoost training for next-day direction prediction."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from xgboost import XGBClassifier

from packages.shared.db import upsert_ignore
from packages.shared.logging import get_logger
from packages.shared.models import Prediction
from services.model.features import FEATURE_COLUMNS, build_feature_matrix
from services.model.metrics import accuracy, log_loss

log = get_logger(__name__)

MODELS_DIR = Path("models")
DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "eval_metric": "logloss",
}


@dataclass
class FoldResult:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    n_train: int
    n_val: int
    log_loss: float
    accuracy: float


@dataclass
class TrainResult:
    model_version: str
    artifact_path: Path
    n_folds: int
    n_predictions_written: int
    folds: list[FoldResult] = field(default_factory=list)

    @property
    def mean_accuracy(self) -> float:
        if not self.folds:
            return 0.0
        return float(np.mean([f.accuracy for f in self.folds]))

    @property
    def mean_log_loss(self) -> float:
        if not self.folds:
            return 0.0
        return float(np.mean([f.log_loss for f in self.folds]))


def _walk_forward_folds(
    dates: np.ndarray, train_window: int, val_window: int, step: int
) -> list[tuple[int, int, int]]:
    """Return list of (train_start_idx, train_end_idx, val_end_idx) on unique sorted dates."""
    folds: list[tuple[int, int, int]] = []
    train_start = 0
    while train_start + train_window + val_window <= len(dates):
        train_end = train_start + train_window
        val_end = train_end + val_window
        folds.append((train_start, train_end, val_end))
        train_start += step
    return folds


def _fit_model(X: pd.DataFrame, y: pd.Series, params: dict | None = None) -> XGBClassifier:
    p = {**DEFAULT_PARAMS, **(params or {})}
    model = XGBClassifier(**p)
    model.fit(X, y)
    return model


def _write_predictions(
    session: Session,
    model_version: str,
    val_predictions: pd.DataFrame,
) -> int:
    if val_predictions.empty:
        return 0
    rows = [
        {
            "symbol": str(r.symbol),
            "ts": r.ts.to_pydatetime() if hasattr(r.ts, "to_pydatetime") else r.ts,
            "model_version": model_version,
            "score": float(r.score),
            "label_pred": int(r.label_pred),
        }
        for r in val_predictions.itertuples(index=False)
    ]
    return upsert_ignore(
        session, Prediction.__table__, rows, ["symbol", "ts", "model_version"]
    )


def train(
    session: Session,
    model_version: str,
    since: dt.date | None = None,
    train_window: int = 252,
    val_window: int = 63,
    step: int = 63,
    models_dir: Path | None = None,
    params: dict | None = None,
) -> TrainResult:
    """Run walk-forward CV, write per-fold validation predictions, persist final model."""
    artifact_dir = models_dir or MODELS_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)

    df = build_feature_matrix(session, since=since)
    if df.empty:
        raise RuntimeError("No training data available — run Phase 1 ingestion first.")

    df = df.sort_values("ts").reset_index(drop=True)
    unique_dates = np.sort(df["ts"].unique())

    folds = _walk_forward_folds(unique_dates, train_window, val_window, step)
    if not folds:
        raise RuntimeError(
            f"Not enough data for walk-forward CV "
            f"({len(unique_dates)} days, need >= {train_window + val_window})"
        )

    fold_results: list[FoldResult] = []
    val_preds: list[pd.DataFrame] = []

    for i, (tr_s, tr_e, va_e) in enumerate(folds):
        train_start = pd.Timestamp(unique_dates[tr_s])
        train_end = pd.Timestamp(unique_dates[tr_e])
        val_end = pd.Timestamp(unique_dates[va_e - 1])

        train_mask = (df["ts"] >= train_start) & (df["ts"] < train_end)
        val_mask = (df["ts"] >= train_end) & (df["ts"] <= val_end)

        train_df = df.loc[train_mask]
        val_df = df.loc[val_mask]
        if train_df.empty or val_df.empty:
            log.warning("Fold %d empty (train=%d val=%d) — skipping", i, len(train_df), len(val_df))
            continue

        max_train_ts = train_df["ts"].max()
        min_val_ts = val_df["ts"].min()
        if not max_train_ts < min_val_ts:
            raise RuntimeError(
                f"Leakage detected in fold {i}: max(train_ts)={max_train_ts} "
                f">= min(val_ts)={min_val_ts}"
            )

        X_tr = train_df[FEATURE_COLUMNS]
        y_tr = train_df["target"]
        X_va = val_df[FEATURE_COLUMNS]
        y_va = val_df["target"]

        model = _fit_model(X_tr, y_tr, params=params)
        proba = model.predict_proba(X_va)[:, 1]
        labels = (proba >= 0.5).astype(int)

        fold_results.append(
            FoldResult(
                fold=i,
                train_start=train_start,
                train_end=train_end,
                val_end=val_end,
                n_train=len(train_df),
                n_val=len(val_df),
                log_loss=log_loss(y_va.values, proba),
                accuracy=accuracy(y_va.values, labels),
            )
        )

        val_preds.append(
            pd.DataFrame(
                {
                    "symbol": val_df["symbol"].values,
                    "ts": val_df["ts"].values,
                    "score": proba,
                    "label_pred": labels,
                }
            )
        )
        log.info(
            "Fold %d: train=[%s..%s] val=[%s..%s] n_train=%d n_val=%d acc=%.3f logloss=%.3f",
            i, train_start.date(), train_end.date(), train_end.date(), val_end.date(),
            len(train_df), len(val_df), fold_results[-1].accuracy, fold_results[-1].log_loss,
        )

    if not fold_results:
        raise RuntimeError("Walk-forward produced zero usable folds.")

    all_val = pd.concat(val_preds, ignore_index=True) if val_preds else pd.DataFrame()
    n_written = _write_predictions(session, model_version, all_val)

    final_model = _fit_model(df[FEATURE_COLUMNS], df["target"], params=params)
    artifact_path = artifact_dir / f"{model_version}.joblib"
    joblib.dump(
        {"model": final_model, "feature_columns": FEATURE_COLUMNS, "model_version": model_version},
        artifact_path,
    )
    log.info(
        "Saved model artifact %s (mean acc=%.3f, mean logloss=%.3f over %d folds)",
        artifact_path,
        float(np.mean([f.accuracy for f in fold_results])),
        float(np.mean([f.log_loss for f in fold_results])),
        len(fold_results),
    )

    return TrainResult(
        model_version=model_version,
        artifact_path=artifact_path,
        n_folds=len(fold_results),
        n_predictions_written=n_written,
        folds=fold_results,
    )
