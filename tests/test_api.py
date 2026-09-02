"""API-level integration test using httpx.AsyncClient + the full FastAPI app.

Overriding the LLM at app.state level lets us test the full HTTP path
(routing, validation, middleware, error handling) without Ollama.
"""
from __future__ import annotations

from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from comparables.main import create_app
from comparables.schemas.mandate import FilterSpec, ParsedMandate, SearchPlan
from comparables.services.workflow_service import WorkflowService


class FakeLLM:
    """Canned LLM used at the app.state level."""

    def __init__(self) -> None:
        self.model = "fake"

    async def complete_json(self, **kwargs):
        schema_model = kwargs.get("schema_model")
        name = schema_model.__name__ if schema_model else "?"
        ctx = kwargs.get("ctx")
        if ctx is not None:
            ctx.inc_llm(10, 20)
        if name == "ParsedMandate":
            return ParsedMandate(
                intent="Find AI fintech in Nordics",
                filters=FilterSpec(
                    industries=["Fintech"],
                    locations=["Finland", "Sweden", "Norway"],
                    employee_min=100,
                    keywords=["AI", "fintech"],
                ),
            )
        if name == "SearchPlan":
            return SearchPlan(use_bm25=True, use_filters=True, limit_per_iter=50)
        if name == "BatchVerdicts":
            from pydantic import BaseModel, Field

            class Verdict(BaseModel):
                company_id: int
                relevant: bool
                evidence_spans: list[str] = Field(default_factory=list)
                reason: str = ""

            class BatchVerdicts(BaseModel):
                verdicts: list[Verdict] = Field(default_factory=list)

            return BatchVerdicts(
                verdicts=[
                    Verdict(
                        company_id=1,
                        relevant=True,
                        evidence_spans=[
                            "AI-powered platform for fraud detection"
                        ],
                        reason="matches mandate",
                    )
                ]
            )
        raise NotImplementedError(name)

    async def complete_text(self, **kwargs):
        return "", 0, 0

    async def ping(self):
        return True

    async def aclose(self):
        return None


@pytest_asyncio.fixture
async def app_with_fake_llm():
    app: FastAPI = create_app()
    # Trigger lifespan manually via ASGI
    async with _LifespanManager(app):
        # Override the LLM with a fake
        app.state.llm_client = FakeLLM()
        # Reconstruct workflow_service with the fake LLM
        app.state.workflow_service = WorkflowService(
            settings=app.state.settings,
            company_repo=app.state.company_repo,
            bm25_repo=app.state.bm25_repo,
            llm=app.state.llm_client,  # type: ignore[arg-type]
            run_repo=app.state.run_repo,
        )
        yield app


# Tiny lifespan helper to avoid pulling in starlette's TestClient (sync).
from contextlib import asynccontextmanager


@asynccontextmanager
async def _LifespanManager(app: FastAPI):
    async with app.router.lifespan_context(app):
        yield


@pytest_asyncio.fixture
async def client(app_with_fake_llm):
    transport = ASGITransport(app=app_with_fake_llm)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ─── Tests ─────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_health_live(client):
    r = await client.get("/api/v1/health/live")
    assert r.status_code == 200
    assert r.json()["status"] == "alive"


@pytest.mark.asyncio
async def test_health_ready(client):
    r = await client.get("/api/v1/health/ready")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ready"] is True
    assert body["checks"]["bm25_loaded"] is True
    assert body["checks"]["sqlite_ready"] is True


@pytest.mark.asyncio
async def test_search_happy_path(client):
    r = await client.post(
        "/api/v1/search",
        json={"query": "Find AI-driven fintech in the Nordics with more than 100 employees"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["run_id"]
    assert body["mandate"]["filters"]["industries"] == ["Fintech"]
    assert body["mandate"]["filters"]["employee_min"] == 100
    assert body["llm_calls"] >= 1
    assert body["llm_calls"] <= 5
    assert body["iterations"] <= 2
    assert len(body["final"]) <= 10
    # First candidate should be Nordic Fintech Solutions
    if body["final"]:
        assert body["final"][0]["company"]["id"] == 1
        assert "AI-powered" in body["final"][0]["evidence"][0]["span"]


@pytest.mark.asyncio
async def test_search_empty_query_rejected(client):
    r = await client.post("/api/v1/search", json={"query": ""})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_search_missing_query_rejected(client):
    r = await client.post("/api/v1/search", json={})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_run_id_header_on_response(client):
    r = await client.post("/api/v1/search", json={"query": "AI fintech in Finland"})
    assert "X-Run-Id" in r.headers
    rid = r.headers["X-Run-Id"]
    # And it should be readable back
    r2 = await client.get(f"/api/v1/runs/{rid}")
    assert r2.status_code == 200
    log = r2.json()
    types = {e["type"] for e in log["events"]}
    assert "run_start" in types
    assert "parse_mandate" in types
    assert "run_end" in types


@pytest.mark.asyncio
async def test_runs_unknown_id_returns_404(client):
    r = await client.get("/api/v1/runs/does_not_exist_123")
    assert r.status_code == 404
