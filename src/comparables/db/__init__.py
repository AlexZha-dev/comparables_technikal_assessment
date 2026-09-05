"""Database layer.

Two roles:

* `db.base.Base` — declarative root for SQLAlchemy ORM models. All models
  inherit from this so a single `Base.metadata` powers autogenerate and
  Alembic revisions.
* `db.session.Database` — async engine + session factory lifecycle.
  One instance per process, owned by `runtime/services.py`. Repositories
  borrow sessions via `db.session()` async context manager.
* `db.models.Company` — the catalog ORM model.
"""
from comparables.db.base import Base
from comparables.db.models.company import Company
from comparables.db.session import Database, get_database, session_scope

__all__ = [
    "Base",
    "Company",
    "Database",
    "get_database",
    "session_scope",
]
