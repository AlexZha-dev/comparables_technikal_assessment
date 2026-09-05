"""CompanyRepository — typed async access to the catalog via SQLAlchemy.

Responsibilities:
    * Read single rows / batches by id.
    * Apply structured filters (industry/location/employee range / founded
      year / revenue bucket) and return typed `Hit` rows for the agent.
    * Count totals (used by `/api/v1/health/ready` and tests).
    * `seed()` for one-shot ingestion (data only — schema is owned by
      Alembic).

The repository is *not* concerned with engine lifecycle. It borrows an
`AsyncSession` from `db.session.Database` via the `session_factory`
attribute on `Database`. Callers manage the session via `async with`.

Schema is owned by `migrations/versions/0001_initial.py`; the ORM model
that mirrors it lives in `db/models/company.py`. If you change the model,
add an Alembic revision — never change the DB schema by hand.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import Select, delete, func, select

from comparables.core.exceptions import CompanyNotFoundError
from comparables.db.models.company import Company
from comparables.db.session import Database
from comparables.schemas.company import CompanyRecord
from comparables.schemas.mandate import FilterSpec
from comparables.schemas.search import Hit


class CompanyRepository:
    """Async typed access to the `companies` table."""

    def __init__(self, db: Database) -> None:
        self._db = db

    # ─── Lifecycle ──────────────────────────────────────────────────────
    async def connect(self) -> None:
        """Bring up the engine if it isn't already. Idempotent."""
        if self._db.session_factory is None:
            await self._db.startup()

    async def close(self) -> None:
        """Repository doesn't own the engine — `db.shutdown()` does.

        We keep this method so callers using the old `repo.close()` pattern
        (runtime/services.py, scripts/ingest.py) keep working. It's a no-op
        for the underlying engine unless no one else holds it.
        """
        # Intentionally NOT calling db.shutdown() — the runtime/services
        # lifecycle owns that. Tests can call `await db.shutdown()` directly.

    # ─── Reads ──────────────────────────────────────────────────────────
    async def fetch_by_ids(self, ids: list[int]) -> list[CompanyRecord]:
        if not ids:
            return []
        async with self._db.session() as session:
            stmt = select(Company).where(Company.id.in_(ids))
            rows = (await session.execute(stmt)).scalars().all()
            return [c.to_record() for c in rows]

    async def fetch_one(self, company_id: int) -> CompanyRecord:
        async with self._db.session() as session:
            company = await session.get(Company, company_id)
            if company is None:
                raise CompanyNotFoundError(f"company_id={company_id} not found")
            return company.to_record()

    async def search_by_filters(
        self, f: FilterSpec, top_k: int = 100
    ) -> list[Hit]:
        """Structured-filter search. Returns typed `Hit(company_id, score)`
        rows with score=1.0 (uniform — ranking by structured match only),
        up to `top_k`.
        """
        stmt = self._eligible_ids_query(f).limit(max(1, min(top_k, 100)))
        async with self._db.session() as session:
            ids = (await session.execute(stmt)).scalars().all()
            return [Hit(company_id=int(cid), score=1.0) for cid in ids]

    async def eligible_ids(self, f: FilterSpec) -> list[int]:
        """Internal eligibility mask: no top-K before lexical ranking.

        This list stays inside the retrieval service, never in agent state or
        a tool response. The returned candidate pool is bounded separately.
        """
        async with self._db.session() as session:
            ids = (await session.execute(self._eligible_ids_query(f))).scalars().all()
            return [int(cid) for cid in ids]

    @staticmethod
    def _eligible_ids_query(f: FilterSpec) -> Select:
        stmt = select(Company.id).order_by(Company.id)
        if f.industries:
            stmt = stmt.where(Company.industry.in_(f.industries))
        if f.locations:
            stmt = stmt.where(Company.location.in_(f.locations))
        if f.revenue_buckets:
            stmt = stmt.where(Company.revenue_range.in_(f.revenue_buckets))
        if f.employee_min is not None:
            stmt = stmt.where(Company.employee_count >= f.employee_min)
        if f.employee_max is not None:
            stmt = stmt.where(Company.employee_count <= f.employee_max)
        if f.founded_after is not None:
            stmt = stmt.where(Company.founded_year >= f.founded_after)
        if f.founded_before is not None:
            stmt = stmt.where(Company.founded_year <= f.founded_before)

        return stmt

    async def count_total(self) -> int:
        async with self._db.session() as session:
            result = await session.execute(select(func.count(Company.id)))
            return int(result.scalar_one() or 0)

    # ─── Ingestion helpers ──────────────────────────────────────────────
    async def seed(self, records: list[CompanyRecord]) -> None:
        """Truncate + bulk insert. Assumes the schema already exists
        (i.e. `alembic upgrade head` has been run).

        Schema management lives in `migrations/versions/`; this method only
        handles the *data* side of ingestion.
        """
        if not records:
            return

        async with self._db.session() as session:
            # Fast truncate. SQLite honors DELETE FROM without WHERE clause.
            await session.execute(delete(Company))

            # Bulk insert via the Core API for speed (50k rows in one round-trip).
            # `mapping` lets us pass dicts keyed by column name.
            payload: list[dict[str, Any]] = [
                {
                    "id": r.id,
                    "name": r.name,
                    "description": r.description,
                    "industry": r.industry,
                    "location": r.location,
                    "founded_year": r.founded_year,
                    "employee_count": r.employee_count,
                    "revenue_range": r.revenue_range,
                }
                for r in records
            ]
            await session.execute(Company.__table__.insert(), payload)
            await session.commit()


__all__ = ["CompanyRepository"]


# Convenience: dump to JSON for debug
def debug_dump(rec: CompanyRecord) -> str:
    import json

    return json.dumps(rec.model_dump(), ensure_ascii=False)
