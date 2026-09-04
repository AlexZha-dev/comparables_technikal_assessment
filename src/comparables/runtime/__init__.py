"""Runtime package: FastAPI plumbing (middleware, lifespan, exception handlers,
service bootstrap) composable into `main.create_app()`.

This package owns the concerns of "how do we turn this into a running ASGI app?".
Domain code (services, repositories, agent) doesn't depend on it.
"""
from __future__ import annotations

from comparables.runtime.exceptions import register_exception_handlers
from comparables.runtime.lifespan import lifespan
from comparables.runtime.middleware import CORSConfig, RunIdMiddleware
from comparables.runtime.services import RuntimeServices, bootstrap_subsystems

__all__ = [
    "CORSConfig",
    "RunIdMiddleware",
    "RuntimeServices",
    "bootstrap_subsystems",
    "lifespan",
    "register_exception_handlers",
]
