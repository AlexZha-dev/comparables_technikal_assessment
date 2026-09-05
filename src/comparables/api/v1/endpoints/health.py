"""Health endpoints: liveness (process up) and readiness (deps OK)."""
from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from comparables.core.config import get_settings

router = APIRouter()


@router.get("/live", summary="Liveness probe")
async def live() -> dict[str, str]:
    """Returns 200 as long as the process is running. Cheap, never fails."""
    return {"status": "alive"}


@router.get("/ready", summary="Readiness probe")
async def ready(request: Request, response: Response) -> dict[str, object]:
    """Report startup readiness flags, requiring data stores but not the LLM.

    - bm25_loaded: BM25 pickle present and parsed.
    - sqlite_ready: SQLite open and responsive.
    - llm_ready: Ollama endpoint reachable.
    """
    settings = get_settings()
    checks = {
        "bm25_loaded": bool(getattr(request.app.state, "bm25_loaded", False)),
        "sqlite_ready": bool(getattr(request.app.state, "sqlite_ready", False)),
        "llm_ready": bool(getattr(request.app.state, "llm_ready", False)),
        "data_dir": settings.paths.sqlite.parent.exists(),
        "runs_dir": settings.paths.runs_dir.exists(),
    }
    # We require data (SQLite + BM25) for the system to be useful. LLM
    # readiness is a soft signal: the workflow supports a limited fallback
    # when Ollama or its configured model is unavailable.
    required = ("bm25_loaded", "sqlite_ready", "data_dir", "runs_dir")
    ok = all(checks[k] for k in required)
    response.status_code = (
        status.HTTP_200_OK if ok else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return {"ready": ok, "checks": checks, "model": settings.llm.model}
