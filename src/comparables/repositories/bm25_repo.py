"""BM25Repository — async wrapper around rank_bm25 BM25Okapi.

Pickle format written by ingestion:
    {
        "bm25": BM25Okapi,           # the index
        "ids": list[int],            # company_id per row, aligned with corpus
        "doc_lens": list[int],       # token count per doc (for stats)
        "avgdl": float,              # average doc length
    }
"""
from __future__ import annotations

import asyncio
import heapq
import pickle
import re
from pathlib import Path
from typing import Any

from comparables.core.exceptions import IndexNotFoundError
from comparables.schemas.search import Hit

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer. Matches ingestion side."""
    return _TOKEN_RE.findall(text.lower())


class BM25Repository:
    """Async read access to the BM25 index over company name+description."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.bm25: Any = None
        self.ids: list[int] = []
        self._id_to_position: dict[int, int] = {}
        self._loaded = False

    # ─── Lifecycle ──────────────────────────────────────────────────────
    async def load(self) -> None:
        if self._loaded:
            return
        if not self.path.exists():
            raise IndexNotFoundError(
                f"BM25 pickle not found at {self.path}; run index-dataset."
            )
        # pickle.load is sync; offload to a thread to avoid blocking the loop.
        data = await asyncio.to_thread(self._load_pickle, self.path)
        self.bm25 = data["bm25"]
        self.ids = list(data["ids"])
        self._id_to_position = {
            company_id: position for position, company_id in enumerate(self.ids)
        }
        self._loaded = True

    @staticmethod
    def _load_pickle(path: Path) -> dict[str, Any]:
        with path.open("rb") as f:
            return pickle.load(f)

    # ─── Search ─────────────────────────────────────────────────────────
    async def search(self, query: str, top_k: int = 50) -> list[Hit]:
        """Return typed `Hit(company_id, score)` list sorted by score desc.

        Empty / no-match queries return [].
        """
        if not self._loaded or self.bm25 is None:
            raise IndexNotFoundError("BM25Repository.load() not called")
        if not query or not query.strip():
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        raw = await asyncio.to_thread(self._search_sync, tokens, top_k)
        return [Hit(company_id=cid, score=float(s)) for cid, s in raw]

    async def score_candidates(
        self,
        query: str,
        company_ids: list[int],
        top_k: int = 100,
    ) -> list[Hit]:
        """Score only an existing eligibility pool without expanding it."""
        if not self._loaded or self.bm25 is None:
            raise IndexNotFoundError("BM25Repository.load() not called")
        tokens = tokenize(query)
        unique_ids = list(dict.fromkeys(company_ids))[:100]
        if not tokens or not unique_ids:
            return []

        def _score() -> list[tuple[int, float]]:
            scores = self.bm25.get_scores(tokens)
            selected = [
                (company_id, float(scores[position]))
                for company_id in unique_ids
                if (position := self._id_to_position.get(company_id)) is not None
                and float(scores[position]) > 0
            ]
            return sorted(selected, key=lambda item: (-item[1], item[0]))[:top_k]

        raw = await asyncio.to_thread(_score)
        return [Hit(company_id=company_id, score=score) for company_id, score in raw]

    async def rank_eligible(
        self, query: str, company_ids: list[int], top_k: int = 100
    ) -> list[Hit]:
        """Exact lexical top-K over the entire SQL eligibility mask.

        Include zero-score eligible rows so the graph can diagnose weak
        retrieval and revise keywords. Missing index IDs are not fabricated.
        """
        if not self._loaded or self.bm25 is None:
            raise IndexNotFoundError("BM25Repository.load() not called")
        tokens = tokenize(query)
        if not company_ids or not tokens:
            return []
        limit = max(1, min(top_k, 100))

        def _rank() -> list[Hit]:
            scores = self.bm25.get_scores(tokens)
            eligible = (
                (cid, max(0.0, float(scores[self._id_to_position[cid]])))
                for cid in set(company_ids) if cid in self._id_to_position
            )
            best = heapq.nsmallest(limit, eligible, key=lambda item: (-item[1], item[0]))
            return [Hit(company_id=cid, score=score) for cid, score in best]

        return await asyncio.to_thread(_rank)

    def _search_sync(self, tokens: list[str], top_k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokens)
        # Argpartition for top-k faster than full sort on large corpora
        # Negative scores for min-heap
        idx_scores = [(i, float(s)) for i, s in enumerate(scores) if s > 0]
        if not idx_scores:
            return []
        top = heapq.nlargest(min(top_k, len(idx_scores)), idx_scores, key=lambda x: x[1])
        return [(self.ids[i], s) for i, s in top]


__all__ = ["BM25Repository", "tokenize"]
