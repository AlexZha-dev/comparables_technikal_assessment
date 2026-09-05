"""GET /api/v1/runs/{run_id} — fetch a stored run log for debugging."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from comparables.api.deps import run_repo_dep
from comparables.repositories.run_repo import RunRepository
from comparables.schemas.run import RunLog

router = APIRouter()


@router.get(
    "/{run_id}",
    response_model=RunLog,
    summary="Fetch a past run's events and metrics",
)
async def get_run(
    run_id: str,
    repo: Annotated[RunRepository, Depends(run_repo_dep)],
) -> RunLog:
    log = await repo.read(run_id)
    if log is None:
        raise HTTPException(status_code=404, detail=f"run_id={run_id} not found")
    return log
