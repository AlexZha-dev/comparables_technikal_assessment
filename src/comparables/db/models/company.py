"""ORM model: `Company` row.

Mirrors the migration-managed schema (migrations/versions/0001_initial.py):
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    industry TEXT NOT NULL,
    location TEXT NOT NULL,
    founded_year INTEGER NOT NULL,
    employee_count INTEGER NOT NULL,
    revenue_range TEXT NOT NULL

Indexes (created in the migration, declared here so SQLAlchemy metadata
matches the live DB for autogenerate diffs):
    idx_company_industry       ON industry
    idx_company_location       ON location
    idx_company_revenue        ON revenue_range
    idx_company_employee_count ON employee_count
    idx_company_founded_year   ON founded_year

The model uses `Mapped[...]` annotations (PEP 484 + SQLAlchemy 2.x
idiomatic style). Don't hand-write `Column(...)` — it bypasses the
type-checking that makes the migration-vs-model diff reliable.
"""
from __future__ import annotations

from sqlalchemy import Index, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from comparables.db.base import Base
from comparables.schemas.company import CompanyRecord


class Company(Base):
    """One row in the catalog. Used at runtime for typed reads."""

    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    industry: Mapped[str] = mapped_column(Text, nullable=False)
    location: Mapped[str] = mapped_column(Text, nullable=False)
    founded_year: Mapped[int] = mapped_column(Integer, nullable=False)
    employee_count: Mapped[int] = mapped_column(Integer, nullable=False)
    revenue_range: Mapped[str] = mapped_column(Text, nullable=False)

    # Index metadata mirrors the migration. `create_index` calls in the
    # migration must match these definitions; if they drift, autogenerate
    # will flag a diff on the next `alembic revision --autogenerate`.
    __table_args__ = (
        Index("idx_company_industry", "industry"),
        Index("idx_company_location", "location"),
        Index("idx_company_revenue", "revenue_range"),
        Index("idx_company_employee_count", "employee_count"),
        Index("idx_company_founded_year", "founded_year"),
    )

    def to_record(self) -> CompanyRecord:
        """Project to the public Pydantic DTO (single source of truth at the API boundary)."""
        return CompanyRecord(
            id=self.id,
            name=self.name,
            description=self.description,
            industry=self.industry,
            location=self.location,
            founded_year=self.founded_year,
            employee_count=self.employee_count,
            revenue_range=self.revenue_range,
        )

    @classmethod
    def from_record(cls, rec: CompanyRecord) -> Company:
        """Hydrate from the public DTO (used by ingest)."""
        return cls(
            id=rec.id,
            name=rec.name,
            description=rec.description,
            industry=rec.industry,
            location=rec.location,
            founded_year=rec.founded_year,
            employee_count=rec.employee_count,
            revenue_range=rec.revenue_range,
        )


__all__ = ["Company"]
