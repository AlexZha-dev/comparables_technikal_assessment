"""Tests for the tool registry and default tools (BM25, filter, get_company)."""
from __future__ import annotations

import pytest
import pytest_asyncio

from comparables.core.config import get_settings
from comparables.core.context import RunContext
from comparables.db.session import Database
from comparables.repositories.bm25_repo import BM25Repository
from comparables.repositories.company_repo import CompanyRepository
from comparables.tools.registry import default_registry


# ─── Fixtures ──────────────────────────────────────────────────────────
@pytest_asyncio.fixture
async def ctx_with_data():
    # Paths are isolated by the shared catalog fixture.
    db = Database.from_settings()
    await db.startup()
    cr = CompanyRepository(db=db)
    await cr.connect()
    bm = BM25Repository(path=get_settings().paths.bm25_pickle)
    await bm.load()

    ctx = RunContext.new()

    class _Repos:
        pass

    repos = _Repos()
    repos.company = cr
    repos.bm25 = bm
    ctx.repos = repos
    ctx.company_repo = cr
    ctx.bm25_repo = bm

    yield ctx
    await cr.close()
    await db.shutdown()


# ─── Registry ──────────────────────────────────────────────────────────
def test_default_registry_has_bounded_search_tools():
    reg = default_registry()
    assert reg.names() == ["bm25_search", "filter_search", "filtered_search", "get_company"]


def test_registry_specs_have_schemas():
    reg = default_registry()
    for s in reg.specs():
        assert s["name"]
        assert s["description"]
        assert s["input_schema"]["type"] == "object"


def test_registry_unknown_tool_raises():
    import asyncio

    from comparables.core.exceptions import RetrievalError

    reg = default_registry()
    ctx = RunContext.new()

    async def go():
        return await reg.invoke("nope", {}, ctx)

    with pytest.raises(RetrievalError):
        asyncio.run(go())


# ─── bm25_search tool ──────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_bm25_search_returns_hits(ctx_with_data):
    reg = default_registry()
    res = await reg.invoke(
        "bm25_search", {"query": "fintech fraud detection", "top_k": 5}, ctx_with_data
    )
    assert res.ok
    assert isinstance(res.data, list)
    assert res.data
    assert all("company_id" in h and "score" in h for h in res.data)


@pytest.mark.asyncio
async def test_bm25_search_empty_query(ctx_with_data):
    reg = default_registry()
    # Pydantic input validation rejects `query=""` (min_length=1) before the
    # tool body even runs — the wrapper turns this into ok=False + error,
    # so the workflow can degrade cleanly without crashing.
    res = await reg.invoke("bm25_search", {"query": ""}, ctx_with_data)
    assert not res.ok
    assert res.error and "string_too_short" in res.error


# ─── filter_search tool ───────────────────────────────────────────────
@pytest.mark.asyncio
async def test_filter_search_fintech_finland_100emp(ctx_with_data):
    reg = default_registry()
    res = await reg.invoke(
        "filter_search",
        {
            "industries": ["Fintech"],
            "locations": ["Finland"],
            "employee_min": 100,
            "top_k": 20,
        },
        ctx_with_data,
    )
    assert res.ok
    assert res.data
    assert len(res.data) <= 20


# ─── get_company tool ──────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_get_company_known_id(ctx_with_data):
    reg = default_registry()
    res = await reg.invoke("get_company", {"company_id": 1}, ctx_with_data)
    assert res.ok
    assert res.data["id"] == 1
    assert "Nordic Fintech" in res.data["name"]
