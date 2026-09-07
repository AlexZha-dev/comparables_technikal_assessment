"""Alembic env.

Reads the database URL from `comparables.core.config.Settings.paths.sqlite`
so the same `.env` / `PATHS__SQLITE` override that the app uses drives
migrations too. One source of truth, no hard-coded paths.

Why sync engine for migrations:
    alembic's migration runner is synchronous; running it on an async
    engine adds greenlet/event-loop juggling without benefit for a one-shot
    `upgrade head`. We translate the async aiosqlite URL to a sync sqlite3
    one at runtime — same `.sqlite` file, no behaviour drift.

Reference for env.py structure: https://alembic.sqlalchemy.org/en/latest/tutorial.html
"""
from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

from alembic import context

# Make `comparables` importable when alembic is invoked from the repo root
# without `poetry install`. The repo's src/ layout requires this.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from comparables.core.config import get_settings  # noqa: E402
from comparables.db.base import Base  # noqa: E402

# Importing the models registers their tables on `Base.metadata`. The import
# side-effect is the whole reason this module exists at the alembic layer.
from comparables.db.models import company as _company_model  # noqa: E402, F401

# Alembic Config object — gives access to alembic.ini values.
config = context.config

# Configure Python logging from alembic.ini (only if the ini file was found).
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Resolve the database URL from Settings (overrides alembic.ini's placeholder).
# Settings.paths.sqlite is a `Path`; convert to a sync sqlite3 URL.
_settings = get_settings()
_db_path = _settings.paths.sqlite.resolve()
_db_url = f"sqlite:///{_db_path.as_posix()}"
config.set_main_option("sqlalchemy.url", _db_url)

# Wire the ORM metadata so autogenerate can diff model ↔ live DB.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — emit SQL to stdout/file without
    a live DB connection. Useful for review or for environments where the
    application DB is read-only from the migration runner's perspective.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # SQLite doesn't support transactional DDL by default; keep the same
        # transactional mode we use at runtime.
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode against a live sqlite3 connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        # SQLite needs check_same_thread=False when sharing a connection across
        # the alembic runner's internal tasks.
        connect_args={"check_same_thread": False},
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
