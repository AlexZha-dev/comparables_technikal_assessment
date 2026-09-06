"""Filter-aware retrieval composed behind small, replaceable interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from comparables.schemas.mandate import FilterSpec
from comparables.schemas.search import Hit


class EligibilityReader(Protocol):
    async def eligible_ids(self, filters: FilterSpec) -> list[int]: ...


class EligibleRanker(Protocol):
    async def rank_eligible(
        self, query: str, company_ids: list[int], top_k: int = 100
    ) -> list[Hit]: ...


@dataclass(frozen=True)
class FilteredSearchResult:
    hits: list[Hit]
    eligible_count: int


class FilteredSearchService:
    """Keep full eligibility internal; expose only the bounded ranked pool."""

    def __init__(self, catalog: EligibilityReader, ranker: EligibleRanker) -> None:
        self._catalog = catalog
        self._ranker = ranker

    async def search(
        self, query: str, filters: FilterSpec, top_k: int = 100
    ) -> FilteredSearchResult:
        ids = await self._catalog.eligible_ids(filters)
        hits = await self._ranker.rank_eligible(query, ids, top_k=max(1, min(top_k, 100)))
        return FilteredSearchResult(hits=hits, eligible_count=len(ids))
