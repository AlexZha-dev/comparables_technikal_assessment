"""FastAPI app factory + lifespan + middleware + exception handlers.

Pattern: layered + DI. Endpoints are thin; business logic lives in services.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from comparables.api.v1.router import api_v1_router
from comparables.core.config import Settings, get_settings
from comparables.core.context import run_id_var
from comparables.core.exceptions import (
    BudgetExceededError,
    ComparablesError,
    CompanyNotFoundError,
    IndexNotFoundError,
    LLMTimeoutError,
    LLMUnavailableError,
    LLMSchemaError,
    MandateParseError,
    RetrievalError,
    WorkflowTimeoutError,
)
from comparables.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


# ─── Middleware ────────────────────────────────────────────────────────
class RunIdMiddleware(BaseHTTPMiddleware):
    """Assigns a run_id per request and injects it into contextvars + response header."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        rid = request.headers.get("X-Run-Id") or uuid.uuid4().hex[:12]
        token = run_id_var.set(rid)
        try:
            response = await call_next(request)
        finally:
            run_id_var.reset(token)
        response.headers["X-Run-Id"] = rid
        return response


# ─── Lifespan ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    logger.info("startup.begin", env=settings.app_env, model=settings.ollama_model)

    # Ensure data + runs dirs exist
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    settings.bm25_pickle_path.parent.mkdir(parents=True, exist_ok=True)
    settings.runs_dir.mkdir(parents=True, exist_ok=True)

    # Eagerly wire repos and LLM. Failures are non-fatal — /health/ready will
    # report which subsystem is down, and individual requests will surface
    # the proper exception via the handlers.
    from comparables.llm.client import LLMClient
    from comparables.repositories.bm25_repo import BM25Repository
    from comparables.repositories.company_repo import CompanyRepository
    from comparables.repositories.run_repo import RunRepository
    from comparables.services.workflow_service import WorkflowService
    from comparables.tools.registry import default_registry

    company_repo = CompanyRepository(path=settings.sqlite_path)
    bm25_repo = BM25Repository(path=settings.bm25_pickle_path)
    run_repo = RunRepository(runs_dir=settings.runs_dir)
    llm = LLMClient(
        base_url=settings.ollama_base_url,
        api_key=settings.ollama_api_key,
        model=settings.ollama_model,
        timeout_s=settings.ollama_timeout_s,
    )

    # SQLite
    try:
        await company_repo.connect()
        app.state.sqlite_ready = True
        logger.info("startup.sqlite_ok", path=str(settings.sqlite_path))
    except Exception as exc:
        app.state.sqlite_ready = False
        logger.warning("startup.sqlite_failed", error=str(exc))

    # BM25
    try:
        await bm25_repo.load()
        app.state.bm25_loaded = True
        logger.info("startup.bm25_ok", path=str(settings.bm25_pickle_path))
    except Exception as exc:
        app.state.bm25_loaded = False
        logger.warning("startup.bm25_failed", error=str(exc))

    # LLM (ping is best-effort)
    try:
        app.state.llm_ready = bool(await llm.ping())
        if app.state.llm_ready:
            logger.info("startup.llm_ok", model=settings.ollama_model)
        else:
            logger.warning("startup.llm_unreachable", base_url=settings.ollama_base_url)
    except Exception as exc:
        app.state.llm_ready = False
        logger.warning("startup.llm_failed", error=str(exc))

    # Workflow service (constructed; not used until first request)
    registry = default_registry()
    app.state.tool_registry = registry
    app.state.company_repo = company_repo
    app.state.bm25_repo = bm25_repo
    app.state.run_repo = run_repo
    app.state.llm_client = llm
    app.state.workflow_service = WorkflowService(
        settings=settings,
        company_repo=company_repo,
        bm25_repo=bm25_repo,
        llm=llm,
        run_repo=run_repo,
    )

    logger.info(
        "startup.done",
        sqlite=app.state.sqlite_ready,
        bm25=app.state.bm25_loaded,
        llm=app.state.llm_ready,
    )
    try:
        yield
    finally:
        try:
            await company_repo.close()
        except Exception:
            pass
        try:
            await llm.aclose()
        except Exception:
            pass
        logger.info("shutdown.complete")


# ─── Exception handlers ───────────────────────────────────────────────
def _err_payload(
    exc: ComparablesError, status: int, request: Request
) -> dict[str, Any]:
    rid = request.headers.get("X-Run-Id") or run_id_var.get()
    return {
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "status": status,
            "run_id": rid,
        }
    }


def register_exception_handlers(app: FastAPI) -> None:
    def _make(status: int):
        async def handler(request: Request, exc: ComparablesError) -> JSONResponse:
            return JSONResponse(_err_payload(exc, status, request), status_code=status)

        return handler

    app.add_exception_handler(IndexNotFoundError, _make(503))
    app.add_exception_handler(CompanyNotFoundError, _make(404))
    app.add_exception_handler(MandateParseError, _make(422))
    app.add_exception_handler(LLMSchemaError, _make(502))
    app.add_exception_handler(LLMUnavailableError, _make(503))
    app.add_exception_handler(LLMTimeoutError, _make(504))
    app.add_exception_handler(BudgetExceededError, _make(429))
    app.add_exception_handler(WorkflowTimeoutError, _make(504))
    app.add_exception_handler(RetrievalError, _make(500))
    app.add_exception_handler(ComparablesError, _make(500))


# ─── App factory ──────────────────────────────────────────────────────
def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a FastAPI app. Optional `settings` override (used in tests)."""
    settings = settings or get_settings()
    configure_logging(settings)

    app = FastAPI(
        title="Comparables.ai Agentic Search",
        version="0.1.0",
        description="Agentic search over a 50k company catalog with grounded evidence.",
        lifespan=lifespan,
    )
    app.state.settings = settings

    # Middleware
    app.add_middleware(RunIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_dev else [],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(api_v1_router, prefix="/api/v1")

    # Handlers
    register_exception_handlers(app)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "service": "comparables-agentic-search",
            "version": app.version,
            "docs": "/docs",
            "api": "/api/v1",
        }

    logger.info("app.created", env=settings.app_env)
    return app


# For `uvicorn comparables.main:create_app --factory`
app = create_app()
