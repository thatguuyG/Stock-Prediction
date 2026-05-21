"""FastAPI dependencies — DB session yielded as request scope."""
from __future__ import annotations

from typing import Iterator

from sqlalchemy.orm import Session

from packages.shared.db import get_sessionmaker


def get_session() -> Iterator[Session]:
    """Open a DB session for the request and close it afterwards.

    Tests override this dependency to use the in-memory SQLite fixture.
    """
    sm = get_sessionmaker()
    session = sm()
    try:
        yield session
    finally:
        session.close()
