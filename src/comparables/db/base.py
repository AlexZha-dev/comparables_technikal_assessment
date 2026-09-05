"""SQLAlchemy declarative base for ORM models.

`Base.metadata` is what Alembic autogenerate uses to diff the live schema
against the model definitions. Adding a new model? Inherit from `Base`.
"""
from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative root. All ORM models in `db.models.*` derive from this."""


__all__ = ["Base"]
