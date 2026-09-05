"""FastAPI exception handlers for the domain exception hierarchy.

Each handler turns an exception into a JSON error body shaped as:

    {"error": {"type": "...", "message": "...", "status": N, "run_id": "..."}}

The HTTP status mapping lives here in one place so it's easy to audit.
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from comparables.core.context import run_id_var
from comparables.core.exceptions import (
    BudgetExceededError,
    CompanyNotFoundError,
    ComparablesError,
    IndexNotFoundError,
    LLMSchemaError,
    LLMTimeoutError,
    LLMUnavailableError,
    MandateParseError,
    RetrievalError,
    WorkflowTimeoutError,
)

# ─── Status mapping ───────────────────────────────────────────────────
# Editable in one place. Key = exception class, value = HTTP status code.
STATUS_BY_EXC: dict[type[ComparablesError], int] = {
    CompanyNotFoundError: 404,
    IndexNotFoundError: 503,
    MandateParseError: 422,
    LLMSchemaError: 502,
    LLMUnavailableError: 503,
    LLMTimeoutError: 504,
    BudgetExceededError: 429,
    WorkflowTimeoutError: 504,
    RetrievalError: 500,
}


def _err_payload(exc: ComparablesError, status: int, request: Request) -> dict[str, Any]:
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
    """Install one handler per exception class in our hierarchy."""

    def _make(status: int):
        async def handler(request: Request, exc: ComparablesError) -> JSONResponse:
            return JSONResponse(_err_payload(exc, status, request), status_code=status)

        return handler

    for exc_type, status in STATUS_BY_EXC.items():
        app.add_exception_handler(exc_type, _make(status))

    # Catch-all: any ComparablesError subclass that wasn't explicitly mapped (or
    # if a future contributor forgets to add it to STATUS_BY_EXC).
    app.add_exception_handler(ComparablesError, _make(500))
