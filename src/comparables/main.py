"""FastAPI app factory — composition only.

All wiring (middleware, lifespan, exception handlers, subsystems) lives in
the `comparables.runtime` package. This module's only job is to assemble
those pieces into a `FastAPI` instance.

Public surface preserved for backwards compatibility:
- `create_app(settings=None)` -> FastAPI
- `app = create_app()` (module-level, for `uvicorn comparables.main:app`)
- `RunIdMiddleware` re-exported from `runtime` so existing imports keep working.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from comparables.api.v1.router import api_v1_router
from comparables.core.config import Settings, get_settings
from comparables.core.logging import configure_logging, get_logger
from comparables.runtime import (
    CORSConfig,
    RunIdMiddleware,
    lifespan,
    register_exception_handlers,
)

logger = get_logger(__name__)


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

    # ─── Middleware ────────────────────────────────────────────────
    # RunId propagates per-request; CORS policy from a typed config.
    app.add_middleware(RunIdMiddleware)
    cors = CORSConfig.for_settings(is_dev=settings.is_dev)
    if cors.allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors.allow_origins,
            allow_credentials=cors.allow_credentials,
            allow_methods=cors.allow_methods,
            allow_headers=cors.allow_headers,
        )

    # ─── Routes ───────────────────────────────────────────────────
    app.include_router(api_v1_router, prefix="/api/v1")

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "service": "comparables-agentic-search",
            "version": app.version,
            "docs": "/docs",
            "api": "/api/v1",
        }

    # ─── Exception handlers ────────────────────────────────────────
    register_exception_handlers(app)

    logger.info("app.created", env=settings.api.env)
    return app


# Module-level ASGI app — `uvicorn comparables.main:app` keeps working.
# `uvicorn comparables.main:create_app --factory` is the recommended form
# (no module-level side effects, settings can be overridden).
app = create_app()


# ─── Backwards-compatible re-exports ─────────────────────────────────
__all__ = ["RunIdMiddleware", "app", "create_app"]
