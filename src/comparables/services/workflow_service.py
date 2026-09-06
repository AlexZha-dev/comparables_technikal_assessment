"""WorkflowService: the single entry point for /api/v1/search.

Builds (or reuses) a LangGraph graph bound to the injected deps, runs it
with a timeout, persists events to RunRepository, and converts the graph
output to a SearchResponse.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from comparables.agent.graph import build_graph
from comparables.core.config import Settings
from comparables.core.context import RunContext
from comparables.core.exceptions import ComparablesError, WorkflowTimeoutError
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
        ctx = RunContext.new(max_llm_calls=s.limits.max_llm_calls_per_run)

        # Wire tools + services onto ctx.repos so tools can find them.
        ctx.repos = SimpleNamespace(company=self._company, bm25=self._bm25)
        ctx.company_repo = self._company
        ctx.bm25_repo = self._bm25

        # Build (or reuse) the graph.
        graph = build_graph(
            ctx=ctx,
            llm=self._llm,
            registry=self._registry,
            company_repo=self._company,
            limits={
                "max_retrieval_iterations": s.limits.max_retrieval_iterations,
                "max_revised_searches": s.limits.max_revised_searches,
                "max_candidates_per_iter": s.limits.max_candidates_per_iter,
                "max_candidates_to_validate": s.limits.max_candidates_to_validate,
                "max_final_results": s.limits.max_final_results,
                "max_llm_calls_per_run": s.limits.max_llm_calls_per_run,
            },
            w_bm=s.scoring.w_bm25,
            w_f=s.scoring.w_filters,
            max_iterations=s.limits.max_retrieval_iterations,
            max_revised=s.limits.max_revised_searches,
            max_validate=s.limits.max_candidates_to_validate,
            max_final=s.limits.max_final_results,
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
                "max_retrieval_iterations": s.limits.max_retrieval_iterations,
                "max_revised_searches": s.limits.max_revised_searches,
                "max_candidates_per_iter": s.limits.max_candidates_per_iter,
                "max_candidates_to_validate": s.limits.max_candidates_to_validate,
                "max_final_results": s.limits.max_final_results,
                "max_llm_calls_per_run": s.limits.max_llm_calls_per_run,
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

        # Run with timeout. Final telemetry is persisted for every exit path.
        final_state: dict[str, Any] = {}
        terminal_error: Exception | None = None
        outcome = "completed"
        try:
            final_state = await asyncio.wait_for(
                graph.ainvoke(initial),  # type: ignore[arg-type]
                timeout=s.limits.timeout_s,
            )
        except TimeoutError as exc:
            terminal_error = WorkflowTimeoutError(
                f"Workflow exceeded {s.limits.timeout_s}s"
            )
            terminal_error.__cause__ = exc
            outcome = "timeout"
            ctx.add_error("workflow", terminal_error)
        except Exception as exc:  # preserve diagnostics before re-raising
            terminal_error = exc
            outcome = "failed"
            ctx.add_error("workflow", exc)
        finally:
            final_items = final_state.get("final") or []
            validation_items = final_state.get("validation") or []
            retrieved_items = final_state.get("candidates") or []
            revisions = sum(
                1 for event in ctx.events if event.get("type") == "revise_search"
            )
            validation_events = [
                event for event in ctx.events if event.get("type") == "validation"
            ]
            retrieval_events = [
                event for event in ctx.events if event.get("type") == "retrieval_iter"
            ]
            last_retrieval = retrieval_events[-1] if retrieval_events else {}
            last_validation = validation_events[-1] if validation_events else {}
            stages = list(
                dict.fromkeys(
                    event["type"]
                    for event in ctx.events
                    if event.get("type")
                    not in {"llm_call", "tool_call", "run_end"}
                )
            )
            estimated_cost = self._estimated_cost_usd(ctx)
            ctx.add_event(
                "run_end",
                outcome=outcome if not ctx.errors else f"{outcome}_with_errors",
                stages=stages,
                final_count=len(final_items),
                retrieved_candidates=(len(retrieved_items) if final_state else int(last_retrieval.get("candidates_after_cap", 0))),
                validated_candidates=(len(validation_items) if final_state else int(last_validation.get("validated", 0))),
                validation_outcome=(
                    validation_events[-1].get("outcome")
                    if validation_events
                    else "not_executed"
                ),
                iterations=int(final_state.get("iteration", last_retrieval.get("iter", 0))),
                revised_searches=revisions,
                llm_calls=ctx.llm_calls,
                tool_calls=ctx.tool_calls,
                tokens_in=ctx.llm_tokens_in,
                tokens_out=ctx.llm_tokens_out,
                estimated_cost_usd=estimated_cost,
                latency_ms=ctx.elapsed_ms(),
                errors=ctx.errors,
            )
            await self._persist_events(ctx)

        if terminal_error is not None:
            if isinstance(terminal_error, ComparablesError):
                terminal_error.run_id = ctx.run_id
            raise terminal_error

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
                    criteria=v.get("criteria", []),
                )
            )

        return SearchResponse(
            run_id=ctx.run_id,
            query=query,
            mandate=final_state.get("original_mandate") or final_state.get("mandate"),
            final=candidates,
            llm_calls=int(ctx.llm_calls or final_state.get("llm_calls", 0)),
            tool_calls=ctx.tool_calls,
            tokens_in=ctx.llm_tokens_in,
            tokens_out=ctx.llm_tokens_out,
            estimated_cost_usd=self._estimated_cost_usd(ctx),
            retrieved_candidates=len(final_state.get("candidates") or []),
            validated_candidates=len(final_state.get("validation") or []),
            revised_search=bool(final_state.get("revised_search_done", False)),
            revised_searches=sum(
                1 for event in ctx.events if event.get("type") == "revise_search"
            ),
            iterations=int(final_state.get("iteration", 0)),
            latency_ms=int(final_state.get("latency_ms", ctx.elapsed_ms())),
            errors=list(final_state.get("errors") or ctx.errors or []),
        )

    async def _persist_events(self, ctx: RunContext) -> None:
        """Best-effort event sink; observability cannot corrupt a valid result."""
        for event in ctx.events:
            try:
                await self._run_repo.write_event(
                    ctx.run_id,
                    event["type"],
                    {**event, "ts": event.get("ts", 0.0)},
                )
            except Exception as exc:
                logger.error(
                    "run_event.persist_failed",
                    run_id=ctx.run_id,
                    event_type=event.get("type"),
                    error=str(exc),
                )

    def _estimated_cost_usd(self, ctx: RunContext) -> float | None:
        """Return actual marginal provider cost when it is knowable."""
        base_url = str(getattr(self._llm, "base_url", "")).casefold()
        if "localhost" in base_url or "127.0.0.1" in base_url or "ollama" in base_url:
            return 0.0
        return None


__all__ = ["WorkflowService"]
