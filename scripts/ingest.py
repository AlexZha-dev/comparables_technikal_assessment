"""CLI: python -m scripts.ingest — builds SQLite + BM25 from companies.json."""
from __future__ import annotations

import asyncio

from comparables.ingestion.index import run_ingestion


def main() -> None:
    stats = asyncio.run(run_ingestion())
    print("Ingestion done:", stats)


if __name__ == "__main__":
    main()
