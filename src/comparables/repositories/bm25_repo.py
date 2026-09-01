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
import pickle
import re
from pathlib import Path
from typing import Any

from comparables.core.exceptions import IndexNotFoundError

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
        self._loaded = True

    @staticmethod
    def _load_pickle(path: Path) -> dict[str, Any]:
        with path.open("rb") as f:
            return pickle.load(f)

    # ─── Search ─────────────────────────────────────────────────────────
    async def search(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        """Return [(company_id, bm25_score), ...] sorted by score desc.

        Empty / no-match queries return [].
        """
        if not self._loaded or self.bm25 is None:
            raise IndexNotFoundError("BM25Repository.load() not called")
        if not query or not query.strip():
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        return await asyncio.to_thread(self._search_sync, tokens, top_k)

    def _search_sync(self, tokens: list[str], top_k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokens)
        # Argpartition for top-k faster than full sort on large corpora
        import heapq

        # Negative scores for min-heap
        idx_scores = [(i, float(s)) for i, s in enumerate(scores) if s > 0]
        if not idx_scores:
            return []
        top = heapq.nlargest(min(top_k, len(idx_scores)), idx_scores, key=lambda x: x[1])
        return [(self.ids[i], s) for i, s in top]


__all__ = ["BM25Repository", "tokenize"]
