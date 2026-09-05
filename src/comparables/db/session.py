"""Async SQLAlchemy engine + session factory.

Scope:
    One `Database` instance per process. It owns the `AsyncEngine` (a
    connection pool) and an `async_sessionmaker`. Call `await db.startup()`
    once at app boot (or ingest start) and `await db.shutdown()` on exit.

    Sessions are short-lived and scoped to a unit of work (one
    repository call, one request). Use `session_scope()` as an async
    context manager:

        async with session_scope() as session:
            ...

    Or borrow the factory directly:

        async with db.session_factory() as session:
            ...

Engine choice:
    SQLite via `aiosqlite`. We could swap to `postgresql+asyncpg` by
    changing one URL — async session API is identical.

Why a class instead of module-level globals:
    Pools need to be disposed explicitly to release connections. A class
    gives us lifecycle hooks (`startup` / `shutdown`) that plug into the
    FastAPI lifespan handler and the ingest CLI cleanly.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from comparables.core.config import Settings, get_settings
from comparables.core.exceptions import IndexNotFoundError

logger = logging.getLogger(__name__)


class Database:
    """Async SQLAlchemy engine + sessionmaker.

    Construct with `Database.from_settings(settings)` to derive the URL from
    `Settings.paths.sqlite`. The engine uses `aiosqlite` under the hood.
    """

    def __init__(self, url: str, *, echo: bool = False) -> None:
        self._url = url
        self.engine: AsyncEngine | None = None
        self.session_factory: async_sessionmaker[AsyncSession] | None = None
        self._echo = echo

    # ─── Construction ──────────────────────────────────────────────────
    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> Database:
        """Build a Database from app settings.

        SQLite URL form for SQLAlchemy async driver:
            sqlite+aiosqlite:///abs/path/to/db.sqlite
        """
        s = settings or get_settings()
        db_path: Path = s.paths.sqlite.resolve()
        url = f"sqlite+aiosqlite:///{db_path.as_posix()}"
        return cls(url, echo=bool(getattr(s.api, "env", "") == "dev"))

    # ─── Lifecycle ─────────────────────────────────────────────────────
    async def startup(self) -> None:
        """Create engine + sessionmaker. Idempotent.

        Raises `IndexNotFoundError` if the SQLite file does not exist yet
        — caller should run alembic migrations / ingest first.
        """
        if self.engine is not None:
            return
        # Path comes from the URL after the third slash.
        # `sqlite+aiosqlite:///abs/path` → `abs/path`.
        prefix = "sqlite+aiosqlite:///"
        if self._url.startswith(prefix):
            db_path = Path(self._url[len(prefix) :])
            if not db_path.exists():
                raise IndexNotFoundError(
                    f"SQLite not found at {db_path}; run migrations / ingest first."
                )

        self.engine = create_async_engine(
            self._url,
            echo=self._echo,
            # SQLite is single-file; a small pool is fine and matches the
            # aiosqlite semantics we'd get without pooling.
            pool_size=5,
            max_overflow=5,
            pool_pre_ping=False,
            future=True,
        )
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            autoflush=False,
        )
        logger.info("db.startup url=%s", self._url)

    async def shutdown(self) -> None:
        if self.engine is None:
            return
        await self.engine.dispose()
        self.engine = None
        self.session_factory = None
        logger.info("db.shutdown")

    # ─── Helpers ───────────────────────────────────────────────────────
    @property
    def url(self) -> str:
        return self._url

    def session(self) -> DatabaseSession:
        """Async context manager that yields a fresh `AsyncSession`.

        Usage:
            async with db.session() as session:
                ...
        """
        if self.session_factory is None:
            raise RuntimeError("Database.startup() not called")
        return DatabaseSession(self.session_factory)


class DatabaseSession:
    """Tiny wrapper so `db.session()` reads naturally.

    Wraps `async_sessionmaker` and commits on clean exit, rolls back on
    exception. Doesn't close on purpose — `async with session_factory()`
    handles that automatically via the context manager protocol.
    """

    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = factory
        self._cm: AsyncSession | None = None

    async def __aenter__(self) -> AsyncSession:
        self._cm = self._factory()
        return await self._cm.__aenter__()

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        assert self._cm is not None
        try:
            await self._cm.__aexit__(exc_type, exc, tb)
        finally:
            self._cm = None


# ─── Module-level singleton (FastAPI-friendly) ────────────────────────
_db_instance: Database | None = None


def get_database() -> Database:
    """Return the process-wide Database, lazily building from settings.

    The instance is shared with `runtime.services.bootstrap_subsystems`
    (which calls `startup()` on it during app boot). For unit tests, build
    a fresh `Database.from_settings(test_settings)` instead and ignore this.
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = Database.from_settings()
    return _db_instance


def reset_database_singleton() -> None:
    """For tests that need to swap settings before constructing the engine."""
    global _db_instance
    _db_instance = None


async def session_scope() -> AsyncIterator[AsyncSession]:
    """Convenience: `async for session in session_scope()`-style usage.

    Equivalent to `async with get_database().session() as session:`.
    """
    db = get_database()
    if db.session_factory is None:
        await db.startup()
    async with db.session() as s:
        yield s


__all__ = ["Database", "DatabaseSession", "get_database", "session_scope"]
