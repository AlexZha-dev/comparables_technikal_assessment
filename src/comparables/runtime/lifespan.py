"""FastAPI lifespan: start/stop the `RuntimeServices` bundle.

Lifespan delegates to `RuntimeServices.startup()` / `.shutdown()` so the
sequence ("connect SQLite, load BM25, ping LLM, build WorkflowService")
lives in one well-tested place rather than being inlined into the ASGI app.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from comparables.core.logging import get_logger
from comparables.runtime.services import bootstrap_subsystems

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = app.state.settings
    logger.info("startup.begin", env=settings.app_env, model=settings.ollama_model)

    # Ensure dirs exist before any repo tries to open them.
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    settings.bm25_pickle_path.parent.mkdir(parents=True, exist_ok=True)
    settings.runs_dir.mkdir(parents=True, exist_ok=True)

    services = bootstrap_subsystems(settings)
    await services.startup()

    # Expose subsystems to Depends() providers in api/deps.py.
    app.state.services = services
    app.state.tool_registry = services.tool_registry
    app.state.company_repo = services.company_repo
    app.state.bm25_repo = services.bm25_repo
    app.state.run_repo = services.run_repo
    app.state.llm_client = services.llm_client
    app.state.workflow_service = services.workflow_service
    app.state.sqlite_ready = services.sqlite_ready
    app.state.bm25_loaded = services.bm25_loaded
    app.state.llm_ready = services.llm_ready

    logger.info(
        "startup.done",
        sqlite=services.sqlite_ready,
        bm25=services.bm25_loaded,
        llm=services.llm_ready,
    )
    try:
        yield
    finally:
        await services.shutdown()
        logger.info("shutdown.complete")
