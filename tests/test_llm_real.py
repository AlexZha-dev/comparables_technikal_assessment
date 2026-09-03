"""Real-Ollama integration test. Skipped automatically if Ollama is unreachable.

Run: pytest tests/test_llm_real.py -v
Or:  python -m scripts.ollama_smoke
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from comparables.core.config import get_settings
from comparables.core.context import RunContext
from comparables.llm.client import LLMClient
from comparables.schemas.mandate import FilterSpec, ParsedMandate, SearchPlan


# ─── Skip if Ollama down ──────────────────────────────────────────────
def _ollama_alive() -> bool:
    s = get_settings()
    try:
        import httpx

        r = httpx.get(f"{s.ollama_base_url.rstrip('/v1')}/api/tags", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_alive(), reason="Ollama not reachable on OLLAMA_BASE_URL"
)


@pytest.fixture
def llm():
    s = get_settings()
    return LLMClient(
        base_url=s.ollama_base_url,
        api_key=s.ollama_api_key,
        model=s.ollama_model,
        timeout_s=s.ollama_timeout_s,
    )


# ─── Tests ────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_ping(llm):
    ok = await llm.ping()
    assert ok, f"Cannot reach Ollama at {llm.base_url}"


@pytest.mark.asyncio
async def test_complete_text(llm):
    text, in_t, out_t = await llm.complete_text(
        system="You are concise.",
        user="Reply with exactly: pong",
        max_tokens=8,
    )
    assert "pong" in text.lower() or "pong" in text.lower(), f"got: {text!r}"
    assert in_t >= 0
    assert out_t >= 0


@pytest.mark.asyncio
async def test_complete_json_mandate(llm):
    ctx = RunContext.new()
    m: ParsedMandate = await llm.complete_json(
        system=(
            "Extract structured fields. Allowed industries: Fintech, Healthcare. "
            "Allowed locations: Finland, Sweden, Germany."
        ),
        user="Find AI fintech in Finland with 200+ employees",
        schema_model=ParsedMandate,
        ctx=ctx,
        max_tokens=300,
    )
    assert m.filters.industries == ["Fintech"], m.filters.industries
    assert "Finland" in m.filters.locations, m.filters.locations
    assert m.filters.employee_min and m.filters.employee_min >= 100
    assert ctx.llm_calls == 1


@pytest.mark.asyncio
async def test_complete_json_plan(llm):
    ctx = RunContext.new()
    m = ParsedMandate(
        intent="test",
        filters=FilterSpec(industries=["Fintech"], locations=["Finland"], keywords=["AI"]),
    )
    plan: SearchPlan = await llm.complete_json(
        system="Decide which tools to use.",
        user=f"Mandate: {m.model_dump_json()}",
        schema_model=SearchPlan,
        ctx=ctx,
        max_tokens=200,
    )
    assert plan.use_bm25 is True
    assert plan.use_filters is True
