"""Regression tests for the workflow's hard correctness invariants."""
from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import ValidationError

from comparables.agent.edges import after_retrieve
from comparables.agent.nodes import (
    _hits_to_scored,
    finalize_node,
    revise_search_node,
    validate_candidates_node,
)
from comparables.agent.policies import EligibilityPolicy
from comparables.agent.sanitize import sanitize_mandate
from comparables.core.context import RunContext
from comparables.core.exceptions import BudgetExceededError, LLMSchemaError, LLMUnavailableError
from comparables.llm.client import LLMClient
from comparables.schemas.company import CompanyRecord
from comparables.schemas.mandate import FilterSpec, ParsedMandate


def _company(**updates) -> CompanyRecord:
    data = {
        "id": 7,
        "name": "Example Mobility",
        "description": "Autonomous driving platform using machine learning.",
        "industry": "Automotive",
        "location": "Netherlands",
        "founded_year": 2020,
        "employee_count": 250,
        "revenue_range": "10M-50M",
    }
    data.update(updates)
    return CompanyRecord(**data)


def test_hybrid_ranking_intersects_mandatory_filter_pool() -> None:
    ranked = _hits_to_scored(
        bm25=[{"company_id": 1, "score": 10.0}, {"company_id": 2, "score": 5.0}],
        filt=[{"company_id": 2, "score": 1.0}, {"company_id": 3, "score": 1.0}],
        w_bm=0.7,
        w_f=0.3,
        require_filter_match=True,
    )
    assert {hit["company_id"] for hit in ranked} == {2, 3}
    assert all(hit["filter_match"] == 1.0 for hit in ranked)


def test_eligibility_policy_checks_every_filter_and_builds_field_evidence() -> None:
    filters = FilterSpec(
        industries=["Automotive"],
        locations=["Netherlands"],
        employee_min=200,
        employee_max=300,
        revenue_buckets=["10M-50M"],
        founded_after=2019,
        founded_before=2021,
    )
    policy = EligibilityPolicy(filters)
    assert policy.matches(_company())
    assert {item.field for item in policy.evidence(_company())} == {
        "industry",
        "location",
        "employee_count",
        "revenue_range",
        "founded_year",
    }
    assert not policy.matches(_company(location="Germany"))


def test_raw_explicit_domain_overrides_generic_llm_inference() -> None:
    mandate = ParsedMandate(
        intent="mobility",
        filters=FilterSpec(
            industries=["Technology"],
            locations=["Netherlands"],
            keywords=["autonomous driving", "machine learning"],
        ),
    )
    normalized = sanitize_mandate(
        mandate,
        "Autonomous driving and machine learning companies in the Netherlands",
    )
    assert normalized.filters.industries == ["Automotive"]


def test_schema_rejects_unexpected_llm_fields() -> None:
    with pytest.raises(ValidationError):
        ParsedMandate.model_validate(
            {"intent": "x", "filters": {}, "must_haves": [], "invented": True}
        )


def test_revision_triggers_for_zero_lexical_overlap_but_not_structured_only() -> None:
    semantic_state = {
        "iteration": 1,
        "revised_search_done": False,
        "mandate": ParsedMandate(
            intent="x",
            filters=FilterSpec(keywords=["renewable energy"]),
        ),
        "candidates": [{"company_id": 1, "score": 0.3, "bm25_score": 0.0}],
    }
    assert after_retrieve(semantic_state, max_iterations=2, max_revised=1) == "revise_search"
    semantic_state["mandate"] = ParsedMandate(
        intent="x",
        filters=FilterSpec(locations=["Germany"]),
    )
    semantic_state["candidates"] = []
    assert after_retrieve(semantic_state, max_iterations=2, max_revised=1) == "validate_candidates"


@pytest.mark.asyncio
async def test_finalize_never_returns_rejected_candidates() -> None:
    context = RunContext.new()
    state = {
        "validation": [
            {
                "company": _company(),
                "score": 0.9,
                "relevant": False,
                "evidence": [],
                "reason": "rejected",
            }
        ]
    }
    result = await finalize_node(state, context, 10)
    assert result["final"] == []


@pytest.mark.asyncio
async def test_llm_retry_budget_is_central_and_hard(monkeypatch) -> None:
    client = LLMClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="fake",
        timeout_s=1,
        max_retries=5,
    )

    async def invalid_call(**kwargs):
        return "not-json", 3, 2

    monkeypatch.setattr(client, "_call", invalid_call)
    context = RunContext.new(max_llm_calls=2)
    with pytest.raises(BudgetExceededError):
        await client.complete_json(
            system="x",
            user="x",
            schema_model=ParsedMandate,
            ctx=context,
            max_attempts=5,
        )
    assert context.llm_calls == 2


def test_direct_run_contexts_do_not_reuse_generated_ids() -> None:
    assert RunContext.new().run_id != RunContext.new().run_id


@pytest.mark.asyncio
async def test_revision_cannot_relax_filters_or_must_haves():
    original = ParsedMandate(
        intent="find mobility", filters=FilterSpec(locations=["Netherlands"], employee_min=201, keywords=["lidar"]),
        must_haves=["machine learning"],
    )

    class RevisingLLM:
        async def complete_json(self, **kwargs):
            return ParsedMandate(intent="relaxed", filters=FilterSpec(locations=["USA"], employee_min=0, keywords=["sensors"]))

    ctx = RunContext.new()
    ctx.repos = SimpleNamespace(company=None)
    state = {"mandate": original, "original_mandate": original, "candidates": [], "raw_query": "mobility", "iteration": 1}
    result = await revise_search_node(state, ctx, RevisingLLM())
    assert result["revised_search_done"]
    assert result["mandate"].filters.locations == ["Netherlands"]
    assert result["mandate"].filters.employee_min == 201
    assert result["mandate"].must_haves == ["machine learning"]
    assert original.filters.keywords == ["lidar"]


@pytest.mark.asyncio
async def test_validation_fallback_uses_original_requirements_after_revision():
    class UnavailableValidator:
        async def complete_json(self, **kwargs):
            raise LLMSchemaError("malformed verdict")

    class Catalog:
        async def fetch_by_ids(self, ids):
            return [_company()]

    original = ParsedMandate(intent="lidar", filters=FilterSpec(keywords=["lidar"]))
    revised = ParsedMandate(intent="mobility", filters=FilterSpec(keywords=["autonomous driving"]))
    result = await validate_candidates_node(
        {"mandate": revised, "original_mandate": original, "raw_query": "lidar", "candidates": [{"company_id": 7, "score": 1.0}]},
        RunContext.new(), UnavailableValidator(), Catalog(), 10,
    )
    assert not result["validation"][0]["relevant"]


@pytest.mark.asyncio
async def test_missing_provider_model_is_a_counted_recoverable_error():
    transport = httpx.MockTransport(lambda request: httpx.Response(404, json={"error": {"message": "model not found"}}))
    client = LLMClient(base_url="http://provider.test/v1", api_key="test", model="absent", timeout_s=1)
    client._client = AsyncOpenAI(base_url=client.base_url, api_key="test", max_retries=0, http_client=httpx.AsyncClient(transport=transport))
    ctx = RunContext.new()
    try:
        with pytest.raises(LLMUnavailableError):
            await client.complete_json(system="test", user="test", schema_model=ParsedMandate, ctx=ctx)
    finally:
        await client.aclose()
    assert ctx.llm_calls == 1
    assert any(event["type"] == "llm_call" and not event["ok"] for event in ctx.events)
