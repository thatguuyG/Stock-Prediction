"""Batch inference for a trained model_version."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy.orm import Session

from packages.shared.db import upsert_ignore
from packages.shared.logging import get_logger
from packages.shared.models import Prediction
from services.model.features import build_feature_matrix
from services.model.train import MODELS_DIR

log = get_logger(__name__)


def load_artifact(model_version: str, models_dir: Path | None = None) -> dict:
    base = models_dir or MODELS_DIR
    path = base / f"{model_version}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact {path} not found — run `stockpred train --model-version {model_version}` first."
        )
    return joblib.load(path)


def predict(
    session: Session,
    model_version: str,
    since: dt.date | None = None,
    models_dir: Path | None = None,
) -> int:
    """Run inference for every row in the feature matrix; upsert into predictions."""
    artifact = load_artifact(model_version, models_dir=models_dir)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    df = build_feature_matrix(session, since=since)
    if df.empty:
        log.warning("predict: feature matrix is empty; nothing to score.")
        return 0

    proba = model.predict_proba(df[feature_columns])[:, 1]
    labels = (proba >= 0.5).astype(int)

    rows = [
        {
            "symbol": str(s),
            "ts": ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
            "model_version": model_version,
            "score": float(p),
            "label_pred": int(label),
        }
        for s, ts, p, label in zip(df["symbol"], df["ts"], proba, labels)
    ]
    inserted = upsert_ignore(
        session, Prediction.__table__, rows, ["symbol", "ts", "model_version"]
    )
    log.info("predict: wrote %d new prediction rows for %s", inserted, model_version)
    return inserted
