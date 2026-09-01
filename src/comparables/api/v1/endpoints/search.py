"""POST /api/v1/search — main agentic search endpoint.

Implemented in Block 8. For now: stub returning 501 with a clear note.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, status

from comparables.api.deps import settings_dep, workflow_service_dep
from comparables.core.config import Settings
from comparables.schemas.search import SearchRequest, SearchResponse
from comparables.services.workflow_service import WorkflowService

router = APIRouter()


@router.post(
    "",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Run an agentic search across the company catalog",
)
async def search_companies(
    body: SearchRequest,
    settings: Settings = Depends(settings_dep),
    workflow: WorkflowService = Depends(workflow_service_dep),
) -> SearchResponse:
    """Interpret the natural-language mandate, retrieve, validate, and return ≤10 grounded hits."""
    return await workflow.invoke(query=body.query, settings=settings)
