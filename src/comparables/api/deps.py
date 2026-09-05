"""FastAPI dependency providers (DI).

Pattern: `get_settings`, `get_db`, `get_llm`, `get_workflow_service`, `get_run_repo`.
All are imported by `api/v1/endpoints/*` and used via `Depends(...)`.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from comparables.core.config import Settings, get_settings
from comparables.llm.client import LLMClient
from comparables.repositories.bm25_repo import BM25Repository
from comparables.repositories.company_repo import CompanyRepository
from comparables.repositories.run_repo import RunRepository
from comparables.services.workflow_service import WorkflowService


def settings_dep() -> Settings:
    """FastAPI wrapper around the cached get_settings()."""
    return get_settings()


def company_repo_dep(request: Request) -> CompanyRepository:
    repo = getattr(request.app.state, "company_repo", None)
    if repo is None:
        from comparables.db.session import Database
        from comparables.repositories.company_repo import CompanyRepository

        db = Database.from_settings(request.app.state.settings)
        repo = CompanyRepository(db=db)
        request.app.state.company_repo = repo
    return repo


def bm25_repo_dep(request: Request) -> BM25Repository:
    repo = getattr(request.app.state, "bm25_repo", None)
    if repo is None:
        from comparables.repositories.bm25_repo import BM25Repository

        repo = BM25Repository(path=request.app.state.settings.paths.bm25_pickle)
        request.app.state.bm25_repo = repo
    return repo


def llm_dep(request: Request) -> LLMClient:
    client = getattr(request.app.state, "llm_client", None)
    if client is None:
        from comparables.llm.client import LLMClient

        client = LLMClient(
            base_url=request.app.state.settings.llm.base_url,
            api_key=request.app.state.settings.llm.api_key,
            model=request.app.state.settings.llm.model,
            timeout_s=request.app.state.settings.llm.timeout_s,
        )
        request.app.state.llm_client = client
    return client


def run_repo_dep(request: Request) -> RunRepository:
    repo = getattr(request.app.state, "run_repo", None)
    if repo is None:
        from comparables.repositories.run_repo import RunRepository

        repo = RunRepository(runs_dir=request.app.state.settings.paths.runs_dir)
        request.app.state.run_repo = repo
    return repo


def workflow_service_dep(
    request: Request,
    settings: Annotated[Settings, Depends(settings_dep)],
    company: Annotated[CompanyRepository, Depends(company_repo_dep)],
    bm25: Annotated[BM25Repository, Depends(bm25_repo_dep)],
    llm: Annotated[LLMClient, Depends(llm_dep)],
    run_repo: Annotated[RunRepository, Depends(run_repo_dep)],
) -> WorkflowService:
    svc = getattr(request.app.state, "workflow_service", None)
    if svc is None:
        from comparables.services.workflow_service import WorkflowService

        svc = WorkflowService(
            settings=settings,
            company_repo=company,
            bm25_repo=bm25,
            llm=llm,
            run_repo=run_repo,
        )
        request.app.state.workflow_service = svc
    return svc
