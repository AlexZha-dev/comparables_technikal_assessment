"""Ingestion: companies.json → SQLite + BM25 pickle.

Run via: python -m scripts.ingest

Order of operations:
    1. `alembic upgrade head`  — ensure schema is current (creates the
       `companies` table + indexes on a fresh DB; no-op if already current).
    2. seed rows into the SQLite file (DELETE + INSERT, not DROP).
    3. build + pickle the BM25 index in parallel-ish (sync, offloaded).

Schema is owned by Alembic (migrations/); this module only handles data.
"""
from __future__ import annotations

import asyncio
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

from rank_bm25 import BM25Okapi

from comparables.core.config import get_settings
from comparables.core.logging import configure_logging, get_logger
from comparables.db.session import Database
from comparables.repositories.bm25_repo import tokenize
from comparables.repositories.company_repo import CompanyRepository
from comparables.schemas.company import CompanyRecord

logger = get_logger(__name__)


def _load_companies(path: Path) -> list[CompanyRecord]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return [CompanyRecord(**item) for item in raw]


def _build_bm25(records: list[CompanyRecord]) -> tuple[BM25Okapi, list[int], list[int], float]:
    """Tokenize once, build BM25Okapi, return index + aligned metadata."""
    tokenized_corpus: list[list[str]] = []
    ids: list[int] = []
    doc_lens: list[int] = []
    for r in records:
        text = f"{r.name} {r.description}"
        toks = tokenize(text)
        tokenized_corpus.append(toks)
        ids.append(r.id)
        doc_lens.append(len(toks))
    bm25 = BM25Okapi(tokenized_corpus)
    avgdl = sum(doc_lens) / max(1, len(doc_lens))
    return bm25, ids, doc_lens, avgdl


def _run_alembic_upgrade() -> None:
    """Bring the schema up to head. Logs under the alembic logger.

    Uses `python -m alembic` so the same venv / interpreter as the rest of
    ingestion is used. `check=True` raises on failure → caller aborts.
    """
    logger.info("ingest.alembic_upgrade")
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stdout:
        for line in result.stdout.splitlines():
            logger.info("alembic", line=line)
    if result.returncode != 0:
        if result.stderr:
            for line in result.stderr.splitlines():
                logger.error("alembic", line=line)
        raise RuntimeError(f"alembic upgrade head failed (rc={result.returncode})")


async def run_ingestion() -> dict[str, int | float]:
    """End-to-end ingestion. Idempotent (overwrites data, schema is migrated)."""
    settings = get_settings()
    configure_logging(settings)

    if not settings.paths.companies_json.exists():
        raise FileNotFoundError(
            f"Source not found: {settings.paths.companies_json}"
        )
    settings.paths.sqlite.parent.mkdir(parents=True, exist_ok=True)
    settings.paths.bm25_pickle.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    logger.info("ingest.load_json", path=str(settings.paths.companies_json))
    records = _load_companies(settings.paths.companies_json)
    t_load = time.perf_counter() - t0
    logger.info("ingest.records_loaded", count=len(records), seconds=round(t_load, 2))

    # Schema migration first — blocks until DB is current.
    t0 = time.perf_counter()
    await asyncio.to_thread(_run_alembic_upgrade)
    t_migrate = time.perf_counter() - t0
    logger.info("ingest.alembic_done", seconds=round(t_migrate, 2))

    # Data seed.
    t0 = time.perf_counter()
    db = Database.from_settings(settings)
    await db.startup()
    repo = CompanyRepository(db=db)
    try:
        await repo.seed(records)
    finally:
        await db.shutdown()
    t_sqlite = time.perf_counter() - t0
    logger.info(
        "ingest.sqlite_seeded",
        path=str(settings.paths.sqlite),
        seconds=round(t_sqlite, 2),
    )

    t0 = time.perf_counter()
    bm25, ids, doc_lens, avgdl = await asyncio.to_thread(_build_bm25, records)
    payload = {
        "bm25": bm25,
        "ids": ids,
        "doc_lens": doc_lens,
        "avgdl": avgdl,
    }
    with settings.paths.bm25_pickle.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    t_bm25 = time.perf_counter() - t0
    logger.info(
        "ingest.bm25_built",
        path=str(settings.paths.bm25_pickle),
        seconds=round(t_bm25, 2),
        avgdl=round(avgdl, 2),
    )

    stats = {
        "records": len(records),
        "sqlite_seconds": round(t_sqlite, 2),
        "bm25_seconds": round(t_bm25, 2),
        "load_seconds": round(t_load, 2),
        "migrate_seconds": round(t_migrate, 2),
    }
    logger.info("ingest.done", **stats)
    return stats


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(run_ingestion())
