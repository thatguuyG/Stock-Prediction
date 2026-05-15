"""Pytest fixtures: SQLite-backed in-memory DB with schema applied.

We use SQLite (rather than pytest-postgresql) so the suite runs without a
running Postgres. The Postgres-specific `ON CONFLICT` upsert paths are
swapped to SQLite's equivalent via SQLAlchemy's dialect detection.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from packages.shared import db as shared_db
from packages.shared.models import Base


@pytest.fixture(name="engine")
def fixture_engine():
    eng = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture(name="session")
def fixture_session(engine, monkeypatch):
    Local = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    monkeypatch.setattr(shared_db, "_engine", engine)
    monkeypatch.setattr(shared_db, "_SessionLocal", Local)

    sess: Session = Local()
    try:
        yield sess
    finally:
        sess.close()
