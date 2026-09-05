"""HTTP middleware: per-request RunId propagation + CORS configuration.

RunIdMiddleware assigns (or reuses) a `run_id` per request, propagates it
through a contextvar so log lines correlate, and echoes it back via the
`X-Run-Id` response header.

CORS configuration is exposed as a small dataclass so tests and main.py don't
accidentally diverge.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from comparables.core.context import run_id_var


# ─── Per-request run_id ────────────────────────────────────────────────
class RunIdMiddleware(BaseHTTPMiddleware):
    """Assigns a run_id per request and injects it into contextvars + response header."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        # Honor an upstream-provided id (so traces stay correlated across services).
        rid = request.headers.get("X-Run-Id") or uuid.uuid4().hex[:12]
        token = run_id_var.set(rid)
        try:
            response = await call_next(request)
        finally:
            run_id_var.reset(token)
        response.headers["X-Run-Id"] = rid
        return response


# ─── CORS policy ───────────────────────────────────────────────────────
@dataclass(frozen=True)
class CORSConfig:
    """Small typed config so CORS isn't scattered as kwargs in main.py."""

    allow_origins: list[str] = field(default_factory=lambda: ["*"])
    allow_methods: list[str] = field(default_factory=lambda: ["*"])
    allow_headers: list[str] = field(default_factory=lambda: ["*"])
    allow_credentials: bool = False

    @classmethod
    def for_settings(cls, *, is_dev: bool, allowed_origins: list[str] | None = None) -> CORSConfig:
        if is_dev:
            # Dev mode: open CORS so OpenAPI/Redoc works from any local tooling.
            return cls(allow_origins=["*"])
        if not allowed_origins:
            # Prod: deny by default. Pass `allowed_origins` explicitly.
            return cls(allow_origins=[])
        return cls(allow_origins=allowed_origins)
