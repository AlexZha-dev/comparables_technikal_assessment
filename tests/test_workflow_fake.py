"""End-to-end workflow test with a FakeLLM (no Ollama needed).

Exercises the full LangGraph path:
  parse_mandate → plan_search → retrieve_and_score → validate_candidates → finalize
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
import pytest_asyncio

from comparables.core.config import get_settings
from comparables.core.exceptions import WorkflowTimeoutError
from comparables.db.session import Database
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
        }

    async def complete_json(self, **kwargs):
        schema_model = kwargs.get("schema_model")
        name = schema_model.__name__ if schema_model else "?"
        ctx = kwargs.get("ctx")
        self.calls.append({"name": name, "kwargs": {k: v for k, v in kwargs.items() if k != "ctx"}})
        if ctx is not None:
            ctx.inc_llm(tokens_in=10, tokens_out=20)
        if name == "BatchVerdicts":
            from comparables.schemas.search import BatchVerdicts
            payload = json.loads(kwargs["user"])
            return BatchVerdicts.model_validate({"verdicts": [
                {"company_id": record["id"], "criteria": [
                    {"criterion_id": criterion["criterion_id"], "status": "supported" if record["id"] == 1 else "insufficient_evidence", "evidence": [{"field": "description", "span": record["description"]}] if record["id"] == 1 else []}
                    for criterion in payload["criteria"]
                ]} for record in payload["companies"]
            ]})
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
    db = Database.from_settings(settings)
    await db.startup()
    cr = CompanyRepository(db=db)
    await cr.connect()
    bm = BM25Repository(path=settings.paths.bm25_pickle)
    await bm.load()
    rr = RunRepository(runs_dir=settings.paths.runs_dir)
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
    await db.shutdown()


@pytest.mark.asyncio
async def test_workflow_end_to_end_with_fake_llm(workflow_svc):
    resp = await workflow_svc.invoke(
        "Find AI-driven fintech companies in the Nordics with more than 100 employees"
    )
    assert resp.run_id
    assert resp.mandate is not None
    assert resp.mandate.filters.industries == ["Fintech"]
    assert resp.mandate.filters.employee_min == 101
    assert resp.llm_calls >= 1
    assert resp.llm_calls <= 5
    assert resp.iterations >= 1
    assert resp.iterations <= 2
    assert len(resp.final) <= 10
    # All returned companies are real
    for c in resp.final:
        assert c.company.id >= 1
        assert c.company.name
        assert c.company.industry == "Fintech"
        assert c.company.location in {"Finland", "Sweden", "Norway"}
        assert c.company.employee_count >= 101
        assert c.relevant and c.evidence
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


@pytest.mark.asyncio
async def test_direct_service_invocations_get_unique_run_ids(workflow_svc):
    first = await workflow_svc.invoke("Find healthcare companies in USA")
    second = await workflow_svc.invoke("Find healthcare companies in USA")
    assert first.run_id != second.run_id


@pytest.mark.asyncio
async def test_timeout_still_persists_terminal_metrics(workflow_svc, tmp_path):
    class SlowLLM:
        model = "slow-fake"
        base_url = "http://localhost:11434/v1"

        async def complete_json(self, **kwargs):
            await asyncio.sleep(0.1)

    settings = workflow_svc._settings.model_copy(
        update={
            "limits": workflow_svc._settings.limits.model_copy(
                update={"timeout_s": 0.01}
            )
        }
    )
    run_repo = RunRepository(tmp_path)
    service = WorkflowService(
        settings=settings,
        company_repo=workflow_svc._company,
        bm25_repo=workflow_svc._bm25,
        llm=SlowLLM(),  # type: ignore[arg-type]
        run_repo=run_repo,
    )

    with pytest.raises(WorkflowTimeoutError) as error:
        await service.invoke("Find fintech in Finland")

    paths = list(tmp_path.glob("*.jsonl"))
    assert len(paths) == 1
    assert error.value.run_id == paths[0].stem
    log = await run_repo.read(paths[0].stem)
    assert log is not None
    assert log.outcome.startswith("timeout")
    assert any(event.type == "run_end" for event in log.events)
