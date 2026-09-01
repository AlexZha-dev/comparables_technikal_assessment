"""API v1 router: aggregates per-resource endpoints with OpenAPI tags."""
from __future__ import annotations

from fastapi import APIRouter

from comparables.api.v1.endpoints import health, runs, search

api_v1_router = APIRouter()
api_v1_router.include_router(health.router, prefix="/health", tags=["health"])
api_v1_router.include_router(search.router, prefix="/search", tags=["search"])
api_v1_router.include_router(runs.router, prefix="/runs", tags=["runs"])
