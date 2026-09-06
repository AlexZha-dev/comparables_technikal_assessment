"""A late eligible ID must beat early low-score IDs without breaking pool limits."""

import pickle
from types import SimpleNamespace

import pytest

from comparables.agent.nodes import retrieve_and_score_node
from comparables.core.context import RunContext
from comparables.db.base import Base
from comparables.db.session import Database
from comparables.ingestion.index import _build_bm25
from comparables.repositories.bm25_repo import BM25Repository
from comparables.repositories.company_repo import CompanyRepository
from comparables.schemas.company import CompanyRecord
from comparables.schemas.mandate import FilterSpec, ParsedMandate, SearchPlan
from comparables.services.filtered_search import FilteredSearchService
from comparables.tools.registry import default_registry


@pytest.mark.asyncio
@pytest.mark.parametrize("keywords", [["quantum underwriting"], []])
async def test_full_eligibility_is_ranked_before_top_k(tmp_path, keywords):
    rows = [
        CompanyRecord(
            id=i,
            name=f"Company {i}",
            description="Ordinary accounting services.",
            industry="Fintech",
            location="Finland",
            founded_year=2020,
            employee_count=200,
            revenue_range="10M-50M",
        )
        for i in range(1, 202)
    ]
    rows[199] = rows[199].model_copy(update={"description": "Quantum underwriting engine."})
    rows[200] = rows[200].model_copy(
        update={"location": "USA", "description": "Quantum underwriting quantum underwriting."}
    )
    db_path = tmp_path / "catalog.sqlite"
    db_path.touch()
    database = Database(f"sqlite+aiosqlite:///{db_path.as_posix()}")
    await database.startup()
    try:
        async with database.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        catalog = CompanyRepository(database)
        await catalog.seed(rows)
        index, ids, lengths, avgdl = _build_bm25(rows)
        path = tmp_path / "bm25.pkl"
        with path.open("wb") as handle:
            pickle.dump({"bm25": index, "ids": ids, "doc_lens": lengths, "avgdl": avgdl}, handle)
        bm25 = BM25Repository(path)
        await bm25.load()
        filters = FilterSpec(locations=["Finland"], keywords=keywords)
        service = FilteredSearchService(catalog, bm25)
        result = await service.search("quantum underwriting", filters, 100)
        assert result.eligible_count == 200
        assert len(result.hits) == 100
        assert result.hits[0].company_id == 200  # previously lost after SQL LIMIT
        assert 201 not in {hit.company_id for hit in result.hits}  # stronger score, wrong country
        assert len({hit.company_id for hit in result.hits}) == 100
        # Ties are deterministic and do not depend on hash/set iteration order.
        assert [hit.company_id for hit in result.hits[1:]] == list(range(1, 100))
        assert not (await service.search("quantum", FilterSpec(locations=["France"]))).hits
        ctx = RunContext.new()
        ctx.repos = SimpleNamespace(company=catalog, bm25=bm25)
        state = {
            "mandate": ParsedMandate(
                intent="underwriting",
                filters=filters,
                semantic_requirements=["quantum underwriting"],
            ),
            "plan": SearchPlan(limit_per_iter=100),
            "iteration": 0,
        }
        output = await retrieve_and_score_node(state, ctx, default_registry(), catalog, 0.7, 0.3)
        assert output["candidates"][0]["company_id"] == 200
        assert len(output["candidates"]) == 100
        assert ctx.tool_calls == 1
        retrieval = next(event for event in ctx.events if event["type"] == "retrieval_iter")
        assert retrieval["eligible_count"] == 200
        assert len(retrieval["candidate_ids"]) == 100
    finally:
        await database.shutdown()
