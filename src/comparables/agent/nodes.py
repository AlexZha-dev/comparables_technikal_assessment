"""LangGraph node functions.

Each node is a small async function (state, ctx) -> dict that returns the
state slice it wants to update. Determinism lives in nodes, not in the graph
itself: every limit is enforced by clamping here.
"""
from __future__ import annotations

import json
import time
from typing import Any

from comparables.agent.prompts import (
    PARSE_MANDATE_SYSTEM,
    PARSE_MANDATE_USER_TEMPLATE,
    PLAN_SEARCH_SYSTEM,
    PLAN_SEARCH_USER_TEMPLATE,
    REVISE_SEARCH_SYSTEM,
    REVISE_SEARCH_USER_TEMPLATE,
    VALIDATE_CANDIDATE_SYSTEM,
    VALIDATE_CANDIDATE_USER_TEMPLATE,
    render_user,
)
from comparables.agent.sanitize import sanitize_mandate
from comparables.agent.state import AgentState, ScoredHit, ValidatedItem
from comparables.core.context import RunContext
from comparables.core.exceptions import (
    BudgetExceededError,
    CompanyNotFoundError,
    LLMSchemaError,
)
from comparables.core.logging import get_logger
from comparables.llm.client import LLMClient
from comparables.schemas.company import CompanyRecord
from comparables.schemas.mandate import ParsedMandate, SearchPlan
from comparables.schemas.search import Evidence
from comparables.tools.registry import ToolRegistry

logger = get_logger(__name__)


# ─── Helpers ───────────────────────────────────────────────────────────
def _limits(state: AgentState) -> dict[str, int]:
    return state.get("settings_limits", {}) or {}


def _hits_to_scored(bm25: list[dict], filt: list[dict], w_bm: float, w_f: float) -> list[ScoredHit]:
    """Merge BM25 and filter hits, normalize BM25 scores, combine."""
    by_id: dict[int, ScoredHit] = {}
    if bm25:
        max_bm = max((h["score"] for h in bm25), default=1.0) or 1.0
        for h in bm25:
            cid = h["company_id"]
            by_id[cid] = {
                "company_id": cid,
                "bm25_score": h["score"] / max_bm,
                "filter_match": 0.0,
                "score": (h["score"] / max_bm) * w_bm,
            }
    for h in filt:
        cid = h["company_id"]
        if cid in by_id:
            by_id[cid]["filter_match"] = 1.0
            by_id[cid]["score"] += 1.0 * w_f
        else:
            by_id[cid] = {
                "company_id": cid,
                "bm25_score": 0.0,
                "filter_match": 1.0,
                "score": 1.0 * w_f,
            }
    return sorted(by_id.values(), key=lambda x: x["score"], reverse=True)


# ─── parse_mandate ─────────────────────────────────────────────────────
async def parse_mandate_node(state: AgentState, ctx: RunContext, llm: LLMClient) -> dict:
    """Convert raw_query to ParsedMandate (1 LLM call)."""
    raw = state.get("raw_query", "")
    user = render_user(PARSE_MANDATE_USER_TEMPLATE, query=raw)
    t0 = time.perf_counter()
    try:
        raw_mandate: ParsedMandate = await llm.complete_json(
            system=PARSE_MANDATE_SYSTEM,
            user=user,
            schema_model=ParsedMandate,
            ctx=ctx,
            temperature=0.0,
            max_tokens=512,
        )
        # Strip hallucinated industries/locations/revenue values.
        mandate = sanitize_mandate(raw_mandate)
        parse_ok = True
        parse_error = None
    except LLMSchemaError as exc:
        logger.warning("parse_mandate.failed", error=str(exc))
        mandate = ParsedMandate(intent=raw, filters=ParsedMandate.model_fields["filters"].default_factory())  # type: ignore[arg-type]
        parse_ok = False
        parse_error = str(exc)

    ctx.add_event(
        "parse_mandate",
        ok=parse_ok,
        duration_ms=int((time.perf_counter() - t0) * 1000),
        mandate=mandate.model_dump() if parse_ok else None,
        error=parse_error,
    )
    return {
        "mandate": mandate,
        "parse_ok": parse_ok,
        "parse_error": parse_error,
        "llm_calls": ctx.llm_calls,
        "tokens_in": ctx.llm_tokens_in,
        "tokens_out": ctx.llm_tokens_out,
    }


# ─── plan_search ───────────────────────────────────────────────────────
async def plan_search_node(state: AgentState, ctx: RunContext, llm: LLMClient) -> dict:
    """Decide which tools and limits to use (1 LLM call). Falls back to defaults."""
    mandate = state.get("mandate")
    if mandate is None:
        return {"plan": None, "plan_ok": False}

    # If mandate is degenerate (no filters, no keywords), skip the plan call.
    f = mandate.filters
    if not (f.keywords or f.industries or f.locations or f.employee_min or f.revenue_buckets):
        return {
            "plan": SearchPlan(use_bm25=True, use_filters=False, limit_per_iter=50),
            "plan_ok": True,
        }

    user = render_user(
        PLAN_SEARCH_USER_TEMPLATE, mandate_json=mandate.model_dump_json()
    )
    t0 = time.perf_counter()
    try:
        plan: SearchPlan = await llm.complete_json(
            system=PLAN_SEARCH_SYSTEM,
            user=user,
            schema_model=SearchPlan,
            ctx=ctx,
            temperature=0.0,
            max_tokens=256,
        )
        plan_ok = True
    except LLMSchemaError:
        plan = SearchPlan(
            use_bm25=bool(f.keywords),
            use_filters=bool(f.industries or f.locations or f.employee_min or f.revenue_buckets),
            limit_per_iter=50,
        )
        plan_ok = False

    ctx.add_event(
        "plan_search",
        ok=plan_ok,
        duration_ms=int((time.perf_counter() - t0) * 1000),
        plan=plan.model_dump(),
    )
    return {"plan": plan, "plan_ok": plan_ok}


# ─── retrieve_and_score ────────────────────────────────────────────────
async def retrieve_and_score_node(
    state: AgentState,
    ctx: RunContext,
    registry: ToolRegistry,
    company_repo: Any,
    w_bm: float,
    w_f: float,
) -> dict:
    """Run planned tools, merge, score, cap at max_candidates_per_iter."""
    limits = _limits(state)
    max_iter = int(limits.get("max_candidates_per_iter", 100))

    mandate = state.get("mandate")
    plan = state.get("plan")
    if mandate is None:
        return {"candidates": [], "iteration": state.get("iteration", 0) + 1}

    f = mandate.filters
    use_bm25 = bool(plan and plan.use_bm25) or bool(f.keywords)
    use_filters = bool(plan and plan.use_filters) or bool(
        f.industries or f.locations or f.employee_min or f.employee_max
        or f.revenue_buckets or f.founded_after or f.founded_before
    )
    per_iter = int((plan.limit_per_iter if plan else 50) or 50)
    per_iter = max(1, min(per_iter, max_iter))

    bm25_hits: list[dict] = []
    filt_hits: list[dict] = []

    if use_bm25 and f.keywords:
        q = " ".join(f.keywords)
        res = await registry.invoke("bm25_search", {"query": q, "top_k": per_iter}, ctx)
        if res.ok:
            bm25_hits = res.data or []

    if use_filters:
        res = await registry.invoke(
            "filter_search",
            {
                "industries": f.industries,
                "locations": f.locations,
                "revenue_buckets": f.revenue_buckets,
                "employee_min": f.employee_min,
                "employee_max": f.employee_max,
                "founded_after": f.founded_after,
                "founded_before": f.founded_before,
                "top_k": per_iter,
            },
            ctx,
        )
        if res.ok:
            filt_hits = res.data or []

    merged = _hits_to_scored(bm25_hits, filt_hits, w_bm, w_f)
    capped = merged[:max_iter]

    ctx.add_event(
        "retrieval_iter",
        iter=state.get("iteration", 0) + 1,
        candidates_in=len(merged),
        candidates_after_cap=len(capped),
        top_score=capped[0]["score"] if capped else 0.0,
        use_bm25=use_bm25,
        use_filters=use_filters,
    )
    return {
        "candidates": capped,
        "iteration": state.get("iteration", 0) + 1,
    }


# ─── revise_search ─────────────────────────────────────────────────────
async def revise_search_node(state: AgentState, ctx: RunContext, llm: LLMClient) -> dict:
    """One-shot: ask LLM to relax mandate (1 LLM call)."""
    mandate = state.get("mandate")
    if mandate is None:
        return {"mandate": mandate}

    # Build top summary
    repo = ctx.repos.company
    top: list[dict] = []
    for h in (state.get("candidates") or [])[:5]:
        try:
            rec = await repo.fetch_one(h["company_id"])
            top.append(
                {
                    "id": rec.id,
                    "name": rec.name,
                    "industry": rec.industry,
                    "location": rec.location,
                    "score": h["score"],
                }
            )
        except CompanyNotFoundError:
            continue

    user = render_user(
        REVISE_SEARCH_USER_TEMPLATE,
        query=state.get("raw_query", ""),
        mandate_json=mandate.model_dump_json(),
        iteration=state.get("iteration", 0),
        top_summary=json.dumps(top, ensure_ascii=False),
    )
    t0 = time.perf_counter()
    try:
        new_raw: ParsedMandate = await llm.complete_json(
            system=REVISE_SEARCH_SYSTEM,
            user=user,
            schema_model=ParsedMandate,
            ctx=ctx,
            temperature=0.3,
            max_tokens=512,
        )
        new_mandate = sanitize_mandate(new_raw)
        ok = True
    except LLMSchemaError as exc:
        logger.warning("revise_search.failed", error=str(exc))
        ok = False
        new_mandate = mandate

    ctx.add_event(
        "revise_search",
        ok=ok,
        duration_ms=int((time.perf_counter() - t0) * 1000),
        iteration=state.get("iteration", 0),
        new_keywords=new_mandate.filters.keywords if ok else None,
    )
    return {
        "mandate": new_mandate,
        "revised_search_done": True,
    }


# ─── validate_candidates ──────────────────────────────────────────────
async def validate_candidates_node(
    state: AgentState,
    ctx: RunContext,
    llm: LLMClient,
    company_repo: Any,
    max_validate: int,
) -> dict:
    """LLM validates top-N candidates against the query (1 LLM call, batched)."""
    limits = _limits(state)
    if ctx.llm_calls >= int(limits.get("max_llm_calls_per_run", 5)):
        # Budget exhausted: fall back to score-based ordering, mark all relevant.
        cands = (state.get("candidates") or [])[:max_validate]
        items: list[ValidatedItem] = []
        for h in cands:
            try:
                rec = await company_repo.fetch_one(h["company_id"])
            except CompanyNotFoundError:
                continue
            items.append(
                {
                    "company": rec,
                    "score": h["score"],
                    "relevant": True,
                    "evidence": [],
                    "reason": "budget_exhausted_no_validation",
                }
            )
        return {"validation": items, "validation_ok": False}

    cands = (state.get("candidates") or [])[:max_validate]
    if not cands:
        return {"validation": [], "validation_ok": True}

    # Hydrate to full records
    by_id: dict[int, CompanyRecord] = {}
    for h in cands:
        try:
            rec = await company_repo.fetch_one(h["company_id"])
            by_id[rec.id] = rec
        except CompanyNotFoundError:
            continue

    # Build a single batched prompt validating all candidates at once.
    # (Bounded by max_validate ≤ 10, so prompt stays small.)
    lines: list[str] = []
    for cid, rec in by_id.items():
        lines.append(
            f"<company id={cid}>\n"
            f"name: {rec.name}\n"
            f"description: {rec.description}\n"
            f"industry: {rec.industry}\n"
            f"location: {rec.location}\n"
            f"employee_count: {rec.employee_count}\n"
            f"revenue_range: {rec.revenue_range}\n"
            f"</company>"
        )
    companies_block = "\n".join(lines)

    from pydantic import BaseModel, Field
    from typing import Literal

    class Verdict(BaseModel):
        company_id: int
        relevant: bool
        evidence_spans: list[str] = Field(default_factory=list)
        reason: str = ""

    class BatchVerdicts(BaseModel):
        verdicts: list[Verdict]

    user = (
        f"Query: {state.get('raw_query', '')}\n\n"
        f"Candidates:\n{companies_block}\n\n"
        f"Return JSON with one verdict per company. "
        f"Every evidence_spans entry MUST be an exact substring of that "
        f"company's name+description text."
    )

    from comparables.llm.structured import schema_instructions as _si

    t0 = time.perf_counter()
    try:
        batch: BatchVerdicts = await llm.complete_json(
            system=VALIDATE_CANDIDATE_SYSTEM
            + "\n\n"
            + _si(BatchVerdicts)
            + "\n\nReturn a `verdicts` list with one entry per company, in the same order.",
            user=user,
            schema_model=BatchVerdicts,
            ctx=ctx,
            temperature=0.0,
            max_tokens=2048,
        )
        ok = True
    except LLMSchemaError as exc:
        logger.warning("validate.failed", error=str(exc))
        batch = BatchVerdicts(verdicts=[])
        ok = False

    # Post-process: enforce substring grounding.
    validated: list[ValidatedItem] = []
    for v in batch.verdicts:
        rec = by_id.get(v.company_id)
        if rec is None:
            continue
        full_text = f"{rec.name} {rec.description}"
        grounded = [s for s in v.evidence_spans if s and s in full_text]
        is_rel = v.relevant and bool(grounded)
        validated.append(
            {
                "company": rec,
                "score": next(
                    (h["score"] for h in cands if h["company_id"] == rec.id), 0.0
                ),
                "relevant": is_rel,
                "evidence": [Evidence(field="name+description", span=s) for s in grounded],
                "reason": v.reason,
            }
        )

    ctx.add_event(
        "validation",
        ok=ok,
        duration_ms=int((time.perf_counter() - t0) * 1000),
        validated=len(validated),
        kept=sum(1 for v in validated if v["relevant"]),
        dropped=sum(1 for v in validated if not v["relevant"]),
    )
    return {"validation": validated, "validation_ok": ok}


# ─── finalize ──────────────────────────────────────────────────────────
async def finalize_node(state: AgentState, ctx: RunContext, max_final: int) -> dict:
    """Sort by score, drop not-relevant, cap at max_final."""
    validation = state.get("validation") or []
    relevant = [v for v in validation if v.get("relevant")]
    if not relevant:
        # Fallback: best-by-score even if validation marked them out.
        relevant = sorted(validation, key=lambda v: v.get("score", 0.0), reverse=True)
    final = sorted(relevant, key=lambda v: v.get("score", 0.0), reverse=True)[:max_final]
    ctx.add_event("run_end", final_count=len(final))
    return {"final": final, "latency_ms": ctx.elapsed_ms()}
