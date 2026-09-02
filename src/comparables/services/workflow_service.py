"""WorkflowService: the single entry point for /api/v1/search.

Builds (or reuses) a LangGraph graph bound to the injected deps, runs it
with a timeout, persists events to RunRepository, and converts the graph
output to a SearchResponse.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from comparables.agent.graph import build_graph
from comparables.core.config import Settings
from comparables.core.context import RunContext
from comparables.core.exceptions import (
    CompanyNotFoundError,
    IndexNotFoundError,
    WorkflowTimeoutError,
)
from comparables.core.logging import get_logger
from comparables.llm.client import LLMClient
from comparables.repositories.bm25_repo import BM25Repository
from comparables.repositories.company_repo import CompanyRepository
from comparables.repositories.run_repo import RunRepository
from comparables.schemas.search import (
    Candidate,
    Evidence,
    SearchResponse,
)
from comparables.tools.registry import ToolRegistry, default_registry

logger = get_logger(__name__)


class WorkflowService:
    """Facade that ties settings + repos + llm + LangGraph into .invoke()."""

    def __init__(
        self,
        settings: Settings,
        company_repo: CompanyRepository,
        bm25_repo: BM25Repository,
        llm: LLMClient,
        run_repo: RunRepository,
    ) -> None:
        self._settings = settings
        self._company = company_repo
        self._bm25 = bm25_repo
        self._llm = llm
        self._run_repo = run_repo
        self._registry: ToolRegistry = default_registry()

    # ─── Public API ────────────────────────────────────────────────────
    async def invoke(self, query: str, settings: Settings | None = None) -> SearchResponse:
        s = settings or self._settings
        ctx = RunContext.new()

        # Wire tools + services onto ctx.repos so tools can find them.
        class _Repos:
            pass

        repos = _Repos()
        repos.company = self._company
        repos.bm25 = self._bm25
        ctx.repos = repos
        ctx.company_repo = self._company
        ctx.bm25_repo = self._bm25

        # Build (or reuse) the graph.
        graph = build_graph(
            ctx=ctx,
            llm=self._llm,
            registry=self._registry,
            company_repo=self._company,
            limits={
                "max_retrieval_iterations": s.max_retrieval_iterations,
                "max_revised_searches": s.max_revised_searches,
                "max_candidates_per_iter": s.max_candidates_per_iter,
                "max_candidates_to_validate": s.max_candidates_to_validate,
                "max_final_results": s.max_final_results,
                "max_llm_calls_per_run": s.max_llm_calls_per_run,
            },
            w_bm=s.score_w_bm25,
            w_f=s.score_w_filters,
            max_iterations=s.max_retrieval_iterations,
            max_revised=s.max_revised_searches,
            max_validate=s.max_candidates_to_validate,
            max_final=s.max_final_results,
        )

        # Persist run_start.
        await self._run_repo.write_event(
            ctx.run_id, "run_start", {"ts": ctx.started_at, "query": query}
        )

        # Initial state.
        initial: dict[str, Any] = {
            "run_id": ctx.run_id,
            "raw_query": query,
            "settings_limits": {
                "max_retrieval_iterations": s.max_retrieval_iterations,
                "max_revised_searches": s.max_revised_searches,
                "max_candidates_per_iter": s.max_candidates_per_iter,
                "max_candidates_to_validate": s.max_candidates_to_validate,
                "max_final_results": s.max_final_results,
                "max_llm_calls_per_run": s.max_llm_calls_per_run,
            },
            "candidates": [],
            "iteration": 0,
            "revised_search_done": False,
            "validation": [],
            "final": [],
            "errors": [],
            "llm_calls": 0,
            "tokens_in": 0,
            "tokens_out": 0,
        }

        # Run with timeout.
        try:
            final_state = await asyncio.wait_for(
                graph.ainvoke(initial),  # type: ignore[arg-type]
                timeout=s.workflow_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise WorkflowTimeoutError(
                f"Workflow exceeded {s.workflow_timeout_s}s"
            ) from exc

        # Persist buffered ctx events.
        for ev in ctx.events:
            await self._run_repo.write_event(
                ctx.run_id, ev["type"], {**ev, "ts": ev.get("ts", 0.0)}
            )

        # Build SearchResponse.
        final = final_state.get("final") or []
        candidates: list[Candidate] = []
        for v in final:
            company = v.get("company")
            if company is None:
                continue
            ev_raw = v.get("evidence", []) or []
            evs: list[Evidence] = []
            for e in ev_raw:
                if isinstance(e, Evidence):
                    evs.append(e)
                elif isinstance(e, dict):
                    evs.append(Evidence(**e))
            candidates.append(
                Candidate(
                    company=company,
                    score=float(v.get("score", 0.0)),
                    relevant=bool(v.get("relevant", True)),
                    evidence=evs,
                    reason=v.get("reason", ""),
                )
            )

        return SearchResponse(
            run_id=ctx.run_id,
            query=query,
            mandate=final_state.get("mandate"),
            final=candidates,
            llm_calls=int(ctx.llm_calls or final_state.get("llm_calls", 0)),
            revised_search=bool(final_state.get("revised_search_done", False)),
            iterations=int(final_state.get("iteration", 0)),
            latency_ms=int(final_state.get("latency_ms", ctx.elapsed_ms())),
            errors=list(final_state.get("errors") or ctx.errors or []),
        )


__all__ = ["WorkflowService"]
