"""SQLAlchemy engine + session factory."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence

from sqlalchemy import create_engine, insert as sa_insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from packages.shared.config import get_settings

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    global _engine  # pylint: disable=global-statement
    if _engine is None:
        _engine = create_engine(get_settings().database_url, pool_pre_ping=True, future=True)
    return _engine


def get_sessionmaker() -> sessionmaker[Session]:
    global _SessionLocal  # pylint: disable=global-statement
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)
    return _SessionLocal


@contextmanager
def session_scope() -> Iterator[Session]:
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def upsert_ignore(
    session: Session,
    table,
    rows: Sequence[dict],
    index_elements: Sequence[str],
) -> int:
    """Insert rows, ignoring conflicts on `index_elements`. Works on Postgres + SQLite."""
    if not rows:
        return 0
    dialect = session.bind.dialect.name if session.bind else session.get_bind().dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert  # pylint: disable=import-outside-toplevel
        stmt = pg_insert(table).values(list(rows)).on_conflict_do_nothing(
            index_elements=list(index_elements)
        )
    elif dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert  # pylint: disable=import-outside-toplevel
        stmt = sqlite_insert(table).values(list(rows)).on_conflict_do_nothing(
            index_elements=list(index_elements)
        )
    else:
        stmt = sa_insert(table).values(list(rows)).prefix_with("OR IGNORE")
    result = session.execute(stmt)
    return result.rowcount or 0


def reset_engine_for_tests(url: str) -> None:
    """Rebind the engine + sessionmaker to a different URL (used by tests)."""
    global _engine, _SessionLocal  # pylint: disable=global-statement
    _engine = create_engine(url, pool_pre_ping=True, future=True)
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
