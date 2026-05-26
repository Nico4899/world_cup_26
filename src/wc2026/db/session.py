"""SQLAlchemy engine + session factory.

Database URL resolution order:
1. Explicit `url` argument to ``get_engine`` / ``get_sessionmaker``.
2. ``DATABASE_URL`` environment variable (matches docker-compose, Fly, scripts,
   and scheduler jobs — see ``src/wc2026/scheduler/jobs.py``).
3. ``WC2026_DATABASE_URL`` environment variable (legacy / explicit override).
4. Default local docker-compose Postgres at port 55432.

All callers should go through ``session_scope()`` for transactional work; the
context manager commits on clean exit and rolls back on exception.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

DEFAULT_LOCAL_URL = "postgresql+psycopg://wc2026:wc2026_local@localhost:55432/wc2026"
ENV_VAR = "WC2026_DATABASE_URL"


def resolve_url(url: str | None = None) -> str:
    if url is not None:
        return url
    return (
        os.environ.get("DATABASE_URL")
        or os.environ.get(ENV_VAR)
        or DEFAULT_LOCAL_URL
    )


@lru_cache(maxsize=8)
def _engine_cached(url: str) -> Engine:
    return create_engine(url, future=True, pool_pre_ping=True)


def get_engine(url: str | None = None) -> Engine:
    """Return a cached SQLAlchemy Engine for the resolved URL."""
    return _engine_cached(resolve_url(url))


def get_sessionmaker(url: str | None = None) -> sessionmaker[Session]:
    """Return a sessionmaker bound to the resolved URL."""
    return sessionmaker(bind=get_engine(url), expire_on_commit=False, future=True)


@contextmanager
def session_scope(url: str | None = None) -> Iterator[Session]:
    """Transactional session context manager. Commits on success, rolls back on error."""
    session = get_sessionmaker(url)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


__all__ = [
    "DEFAULT_LOCAL_URL",
    "ENV_VAR",
    "get_engine",
    "get_sessionmaker",
    "resolve_url",
    "session_scope",
]
