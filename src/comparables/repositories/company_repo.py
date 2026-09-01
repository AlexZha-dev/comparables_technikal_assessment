"""CompanyRepository — async SQLite access to the catalog.

Schema (created in ingestion):
    id              INTEGER PRIMARY KEY
    name            TEXT NOT NULL
    description     TEXT NOT NULL
    industry        TEXT NOT NULL
    location        TEXT NOT NULL
    founded_year    INTEGER NOT NULL
    employee_count  INTEGER NOT NULL
    revenue_range   TEXT NOT NULL

Indexes (also created in ingestion):
    idx_company_industry        ON industry
    idx_company_location        ON location
    idx_company_revenue         ON revenue_range
    idx_company_employee_count  ON employee_count
    idx_company_founded_year    ON founded_year
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiosqlite

from comparables.core.exceptions import CompanyNotFoundError, IndexNotFoundError
from comparables.schemas.company import CompanyRecord
from comparables.schemas.mandate import FilterSpec

_SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    id              INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT NOT NULL,
    industry        TEXT NOT NULL,
    location        TEXT NOT NULL,
    founded_year    INTEGER NOT NULL,
    employee_count  INTEGER NOT NULL,
    revenue_range   TEXT NOT NULL
);
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_company_industry        ON companies(industry);",
    "CREATE INDEX IF NOT EXISTS idx_company_location        ON companies(location);",
    "CREATE INDEX IF NOT EXISTS idx_company_revenue         ON companies(revenue_range);",
    "CREATE INDEX IF NOT EXISTS idx_company_employee_count  ON companies(employee_count);",
    "CREATE INDEX IF NOT EXISTS idx_company_founded_year    ON companies(founded_year);",
]


class CompanyRepository:
    """Async read-only access to the company catalog."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn: aiosqlite.Connection | None = None

    # ─── Lifecycle ──────────────────────────────────────────────────────
    async def connect(self) -> None:
        if self._conn is not None:
            return
        if not self.path.exists():
            raise IndexNotFoundError(f"SQLite not found at {self.path}; run index-dataset.")
        self._conn = await aiosqlite.connect(str(self.path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA synchronous=NORMAL;")
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise IndexNotFoundError("CompanyRepository.connect() not called")
        return self._conn

    # ─── Reads ──────────────────────────────────────────────────────────
    async def fetch_by_ids(self, ids: list[int]) -> list[CompanyRecord]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        cur = await self.conn.execute(
            f"SELECT * FROM companies WHERE id IN ({placeholders})", ids
        )
        rows = await cur.fetchall()
        await cur.close()
        return [self._row_to_record(r) for r in rows]

    async def fetch_one(self, company_id: int) -> CompanyRecord:
        cur = await self.conn.execute("SELECT * FROM companies WHERE id = ?", (company_id,))
        row = await cur.fetchone()
        await cur.close()
        if row is None:
            raise CompanyNotFoundError(f"company_id={company_id} not found")
        return self._row_to_record(row)

    async def search_by_filters(
        self, f: FilterSpec, top_k: int = 100
    ) -> list[tuple[int, float]]:
        """Structured-filter search. Returns (id, score=1.0) up to top_k.

        Score is uniform here; ranking by structured match only.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if f.industries:
            clauses.append(
                "industry IN (" + ",".join("?" * len(f.industries)) + ")"
            )
            params.extend(f.industries)
        if f.locations:
            clauses.append(
                "location IN (" + ",".join("?" * len(f.locations)) + ")"
            )
            params.extend(f.locations)
        if f.revenue_buckets:
            clauses.append(
                "revenue_range IN (" + ",".join("?" * len(f.revenue_buckets)) + ")"
            )
            params.extend(f.revenue_buckets)
        if f.employee_min is not None:
            clauses.append("employee_count >= ?")
            params.append(f.employee_min)
        if f.employee_max is not None:
            clauses.append("employee_count <= ?")
            params.append(f.employee_max)
        if f.founded_after is not None:
            clauses.append("founded_year >= ?")
            params.append(f.founded_after)
        if f.founded_before is not None:
            clauses.append("founded_year <= ?")
            params.append(f.founded_before)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT id FROM companies {where} LIMIT ?"
        params.append(top_k)

        cur = await self.conn.execute(sql, params)
        rows = await cur.fetchall()
        await cur.close()
        return [(int(r[0]), 1.0) for r in rows]

    async def count_total(self) -> int:
        cur = await self.conn.execute("SELECT COUNT(*) FROM companies")
        row = await cur.fetchone()
        await cur.close()
        return int(row[0]) if row else 0

    # ─── Ingestion helpers ──────────────────────────────────────────────
    async def bulk_insert(self, records: list[CompanyRecord]) -> None:
        """Used by ingestion. NOT used at runtime."""
        if self._conn is None:
            self._conn = await aiosqlite.connect(str(self.path))
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA synchronous=OFF;")
        await self._conn.execute("DROP TABLE IF EXISTS companies;")
        await self._conn.executescript(_SCHEMA)
        for stmt in _INDEXES:
            await self._conn.execute(stmt)
        await self._conn.executemany(
            """INSERT INTO companies
               (id, name, description, industry, location, founded_year,
                employee_count, revenue_range)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    r.id,
                    r.name,
                    r.description,
                    r.industry,
                    r.location,
                    r.founded_year,
                    r.employee_count,
                    r.revenue_range,
                )
                for r in records
            ],
        )
        await self._conn.commit()

    # ─── Internal ───────────────────────────────────────────────────────
    @staticmethod
    def _row_to_record(row: aiosqlite.Row) -> CompanyRecord:
        return CompanyRecord(
            id=int(row["id"]),
            name=str(row["name"]),
            description=str(row["description"]),
            industry=str(row["industry"]),
            location=str(row["location"]),
            founded_year=int(row["founded_year"]),
            employee_count=int(row["employee_count"]),
            revenue_range=str(row["revenue_range"]),
        )


__all__ = ["CompanyRepository"]


# Convenience: dump to JSON for debug
def debug_dump(rec: CompanyRecord) -> str:
    return json.dumps(rec.model_dump(), ensure_ascii=False)
