"""Ingestion: companies.json → SQLite + BM25 pickle.

Run via: python -m scripts.ingest
"""
from __future__ import annotations

import asyncio
import json
import pickle
import time
from pathlib import Path

from rank_bm25 import BM25Okapi

from comparables.core.config import get_settings
from comparables.core.logging import configure_logging, get_logger
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


async def run_ingestion() -> dict[str, int | float]:
    """End-to-end ingestion. Idempotent (overwrites). Returns stats."""
    settings = get_settings()
    configure_logging(settings)

    if not settings.companies_json_path.exists():
        raise FileNotFoundError(
            f"Source not found: {settings.companies_json_path}"
        )
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    settings.bm25_pickle_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    logger.info("ingest.load_json", path=str(settings.companies_json_path))
    records = _load_companies(settings.companies_json_path)
    t_load = time.perf_counter() - t0
    logger.info("ingest.records_loaded", count=len(records), seconds=round(t_load, 2))

    t0 = time.perf_counter()
    repo = CompanyRepository(path=settings.sqlite_path)
    try:
        await repo.bulk_insert(records)
    finally:
        await repo.close()
    t_sqlite = time.perf_counter() - t0
    logger.info("ingest.sqlite_built", path=str(settings.sqlite_path), seconds=round(t_sqlite, 2))

    t0 = time.perf_counter()
    bm25, ids, doc_lens, avgdl = await asyncio.to_thread(_build_bm25, records)
    payload = {
        "bm25": bm25,
        "ids": ids,
        "doc_lens": doc_lens,
        "avgdl": avgdl,
    }
    with settings.bm25_pickle_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    t_bm25 = time.perf_counter() - t0
    logger.info(
        "ingest.bm25_built",
        path=str(settings.bm25_pickle_path),
        seconds=round(t_bm25, 2),
        avgdl=round(avgdl, 2),
    )

    stats = {
        "records": len(records),
        "sqlite_seconds": round(t_sqlite, 2),
        "bm25_seconds": round(t_bm25, 2),
        "load_seconds": round(t_load, 2),
    }
    logger.info("ingest.done", **stats)
    return stats


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(run_ingestion())
