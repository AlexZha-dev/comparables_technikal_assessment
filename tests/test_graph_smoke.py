"""Smoke test that the LangGraph graph compiles and topology is sane (no Ollama needed)."""
from __future__ import annotations

from pathlib import Path

import pytest_asyncio

from comparables.agent.graph import build_graph
from comparables.core.context import RunContext
from comparables.repositories.bm25_repo import BM25Repository
from comparables.repositories.company_repo import CompanyRepository
from comparables.tools.registry import default_registry


# A no-op LLM (we don't actually call it in this test)
class _NullLLM:
    model = "stub"
    base_url = "stub"

    async def complete_json(self, **kwargs):
        raise NotImplementedError

    async def complete_text(self, **kwargs):
        raise NotImplementedError

    async def ping(self):
        return True


@pytest_asyncio.fixture
async def graph_built():
    cr = CompanyRepository(path=Path("data/companies.sqlite"))
    bm = BM25Repository(path=Path("data/bm25.pkl"))
    await cr.connect()
    await bm.load()

    ctx = RunContext.new()

    class _Repos:
        pass

    repos = _Repos()
    repos.company = cr
    repos.bm25 = bm
    ctx.repos = repos

    g = build_graph(
        ctx=ctx,
        llm=_NullLLM(),  # type: ignore[arg-type]
        registry=default_registry(),
        company_repo=cr,
        limits={
            "max_retrieval_iterations": 2,
            "max_revised_searches": 1,
            "max_candidates_per_iter": 100,
            "max_candidates_to_validate": 10,
            "max_final_results": 10,
            "max_llm_calls_per_run": 5,
        },
        w_bm=0.7,
        w_f=0.3,
        max_iterations=2,
        max_revised=1,
        max_validate=10,
        max_final=10,
    )
    yield g
    await cr.close()


def test_graph_compiles(graph_built):
    # Just exercising the fixture ensures build_graph() doesn't raise.
    assert graph_built is not None
