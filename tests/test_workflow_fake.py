"""End-to-end workflow test with a FakeLLM (no Ollama needed).

Exercises the full LangGraph path:
  parse_mandate → plan_search → retrieve_and_score → validate_candidates → finalize
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from comparables.core.config import get_settings
from comparables.core.context import RunContext
from comparables.repositories.bm25_repo import BM25Repository
from comparables.repositories.company_repo import CompanyRepository
from comparables.repositories.run_repo import RunRepository
from comparables.schemas.mandate import FilterSpec, ParsedMandate, SearchPlan
from comparables.services.workflow_service import WorkflowService


class FakeLLM:
    """Returns canned Pydantic models in response to schema_model type."""

    def __init__(self) -> None:
        self.model = "fake"
        self.calls: list[dict[str, Any]] = []
        # Map schema_model name → factory producing an instance
        self._answers: dict[str, Any] = {
            "ParsedMandate": lambda: ParsedMandate(
                intent="Find AI fintech in Nordics >100 employees",
                filters=FilterSpec(
                    industries=["Fintech"],
                    locations=["Finland", "Sweden", "Norway"],
                    employee_min=100,
                    keywords=["AI", "fintech"],
                ),
                must_haves=[],
            ),
            "SearchPlan": lambda: SearchPlan(
                use_bm25=True, use_filters=True, limit_per_iter=50
            ),
            "BatchVerdicts": self._verdicts_for,
        }

    def _verdicts_for(self) -> Any:
        from pydantic import BaseModel, Field

        class Verdict(BaseModel):
            company_id: int
            relevant: bool
            evidence_spans: list[str] = Field(default_factory=list)
            reason: str = ""

        class BatchVerdicts(BaseModel):
            verdicts: list[Verdict] = Field(default_factory=list)

        # Top 3 candidates should be relevant with grounded spans
        sample = [
            (1, "Nordic Fintech Solutions", "AI-powered platform for fraud detection, banking analytics, and risk assessment."),
            (2, "Baltic Payments Cloud", "Cloud-native payments infrastructure and financial data platform for digital banking."),
            (476, None, None),  # unknown — should be handled
        ]
        vs = []
        for cid, _name, desc in sample:
            if desc:
                vs.append(
                    Verdict(
                        company_id=cid,
                        relevant=True,
                        evidence_spans=[desc[:30]],  # take a 30-char prefix as a valid substring
                        reason="matches AI fintech profile",
                    )
                )
        return BatchVerdicts(verdicts=vs)

    async def complete_json(self, **kwargs):
        schema_model = kwargs.get("schema_model")
        name = schema_model.__name__ if schema_model else "?"
        ctx = kwargs.get("ctx")
        self.calls.append({"name": name, "kwargs": {k: v for k, v in kwargs.items() if k != "ctx"}})
        if ctx is not None:
            ctx.inc_llm(tokens_in=10, tokens_out=20)
        factory = self._answers.get(name)
        if factory is None:
            raise NotImplementedError(f"no fake answer for {name}")
        return factory()

    async def complete_text(self, **kwargs):
        return "", 0, 0

    async def ping(self):
        return True


@pytest_asyncio.fixture
async def workflow_svc():
    settings = get_settings()
    cr = CompanyRepository(path=Path("data/companies.sqlite"))
    bm = BM25Repository(path=Path("data/bm25.pkl"))
    await cr.connect()
    await bm.load()
    rr = RunRepository(runs_dir=Path("runs"))
    fake = FakeLLM()
    svc = WorkflowService(
        settings=settings,
        company_repo=cr,
        bm25_repo=bm,
        llm=fake,  # type: ignore[arg-type]
        run_repo=rr,
    )
    yield svc
    await cr.close()


@pytest.mark.asyncio
async def test_workflow_end_to_end_with_fake_llm(workflow_svc):
    resp = await workflow_svc.invoke(
        "Find AI-driven fintech companies in the Nordics with more than 100 employees"
    )
    assert resp.run_id
    assert resp.mandate is not None
    assert resp.mandate.filters.industries == ["Fintech"]
    assert resp.mandate.filters.employee_min == 100
    assert resp.llm_calls >= 1
    assert resp.llm_calls <= 5
    assert resp.iterations >= 1
    assert resp.iterations <= 2
    assert len(resp.final) <= 10
    # All returned companies are real
    for c in resp.final:
        assert c.company.id >= 1
        assert c.company.name
    # At least one candidate should be kept (BM25 finds "Nordic Fintech Solutions" easily)
    assert len(resp.final) >= 1


@pytest.mark.asyncio
async def test_workflow_persists_run_log(workflow_svc):
    resp = await workflow_svc.invoke("Find biotech in Germany")
    log = await workflow_svc._run_repo.read(resp.run_id)
    assert log is not None
    types = {e.type for e in log.events}
    assert "run_start" in types
    assert "parse_mandate" in types
    assert "run_end" in types
