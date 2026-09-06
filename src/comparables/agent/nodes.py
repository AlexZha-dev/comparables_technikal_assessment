"""LangGraph node functions.

Each node is a small async function (state, ctx) -> dict that returns the
state slice it wants to update. Determinism lives in nodes, not in the graph
itself: every limit is enforced by clamping here.
"""
from __future__ import annotations

import json
import time
from typing import Any

from comparables.agent.policies import (
    CandidateRanker,
    EligibilityPolicy,
    has_structured_filters,
)
from comparables.agent.prompts import (
    PARSE_MANDATE_SYSTEM,
    PARSE_MANDATE_USER_TEMPLATE,
    REVISE_SEARCH_SYSTEM,
    REVISE_SEARCH_USER_TEMPLATE,
    render_user,
)
from comparables.agent.sanitize import fallback_mandate, sanitize_mandate
from comparables.agent.semantic import retrieval_query, semantic_criteria
from comparables.agent.state import AgentState, ScoredHit, ValidatedItem
from comparables.core.context import RunContext
from comparables.core.exceptions import (
    BudgetExceededError,
    CompanyNotFoundError,
    LLMSchemaError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from comparables.core.logging import get_logger
from comparables.llm.client import LLMClient
from comparables.schemas.company import CompanyRecord
from comparables.schemas.mandate import ParsedMandate, SearchPlan
from comparables.services.semantic_validation import SemanticValidationService
from comparables.tools.registry import ToolRegistry

logger = get_logger(__name__)


# ─── Helpers ───────────────────────────────────────────────────────────
def _limits(state: AgentState) -> dict[str, int]:
    return state.get("settings_limits", {}) or {}


_LLM_RECOVERABLE_ERRORS = (
    LLMSchemaError,
    LLMTimeoutError,
    LLMUnavailableError,
    BudgetExceededError,
)


def _hits_to_scored(
    bm25: list[dict],
    filt: list[dict],
    w_bm: float,
    w_f: float,
    *,
    require_filter_match: bool = False,
    keyword_boost: float = 1.0,
) -> list[ScoredHit]:
    """Compatibility wrapper around the deterministic ranking strategy."""
    return CandidateRanker(w_bm, w_f).merge(
        bm25,
        filt,
        require_filter_match=require_filter_match,
        keyword_boost=keyword_boost,
    )


# ─── parse_mandate ─────────────────────────────────────────────────────
async def parse_mandate_node(state: AgentState, ctx: RunContext, llm: LLMClient) -> dict:
    """Convert raw_query to ParsedMandate (at most two LLM attempts)."""
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
            max_attempts=2,
            stage="parse_mandate",
        )
        mandate = sanitize_mandate(raw_mandate, raw_query=raw)
        parse_ok = True
        parse_error = None
    except _LLM_RECOVERABLE_ERRORS as exc:
        logger.warning("parse_mandate.failed", error=str(exc))
        mandate = fallback_mandate(raw)
        parse_ok = False
        parse_error = str(exc)
        ctx.add_error("parse_mandate", exc)

    ctx.add_event(
        "parse_mandate",
        ok=parse_ok,
        duration_ms=int((time.perf_counter() - t0) * 1000),
        mandate=mandate.model_dump(),
        fallback_used=not parse_ok,
        error=parse_error,
    )
    return {
        "mandate": mandate,
        "original_mandate": mandate.model_copy(deep=True),
        "parse_ok": parse_ok,
        "parse_error": parse_error,
        "llm_calls": ctx.llm_calls,
        "tokens_in": ctx.llm_tokens_in,
        "tokens_out": ctx.llm_tokens_out,
    }


# ─── plan_search ───────────────────────────────────────────────────────
async def plan_search_node(state: AgentState, ctx: RunContext) -> dict:
    """Build a deterministic tool plan from the validated mandate."""
    mandate = state.get("mandate")
    if mandate is None:
        return {"plan": None, "plan_ok": False}

    f = mandate.filters
    use_bm25 = bool(retrieval_query(mandate))
    use_filters = has_structured_filters(f)
    plan = SearchPlan(
        use_bm25=use_bm25,
        use_filters=use_filters,
        keyword_boost=1.0,
        limit_per_iter=100,
        rationale=(
            "filtered top-K over full SQL eligibility"
            if use_bm25 and use_filters
            else "structured SQL eligibility"
            if use_filters
            else "lexical BM25 retrieval"
            if use_bm25
            else "no retrievable criteria"
        ),
    )
    ctx.add_event(
        "plan_search",
        ok=True,
        deterministic=True,
        fallback_parse=not state.get("parse_ok", False),
        plan=plan.model_dump(),
    )
    return {"plan": plan, "plan_ok": True}


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
    hard_filters = has_structured_filters(f)
    lexical_query = retrieval_query(mandate)
    use_bm25 = bool(lexical_query) and (plan is None or plan.use_bm25)
    # An LLM-generated plan can never disable an explicit user constraint.
    use_filters = hard_filters
    per_iter = int((plan.limit_per_iter if plan else 100) or 100)
    per_iter = max(1, min(per_iter, max_iter))

    bm25_hits: list[dict] = []
    filt_hits: list[dict] = []

    eligible_count: int | None = None
    if use_filters and use_bm25:
        res = await registry.invoke(
            "filtered_search",
            {"query": lexical_query, "filters": f.model_dump(), "top_k": per_iter},
            ctx,
        )
        if res.ok:
            hits = res.data or []
            filt_hits = [{"company_id": hit["company_id"], "score": 1.0} for hit in hits]
            bm25_hits = [hit for hit in hits if hit["score"] > 0]
            eligible_count = res.meta.get("eligible_count")
    elif use_filters:
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

    if use_bm25 and not hard_filters:
        bm25_args: dict[str, Any] = {"query": lexical_query, "top_k": per_iter}
        res = await registry.invoke("bm25_search", bm25_args, ctx)
        if res.ok:
            bm25_hits = res.data or []

    merged = _hits_to_scored(
        bm25_hits,
        filt_hits,
        w_bm,
        w_f,
        require_filter_match=hard_filters,
        keyword_boost=float(plan.keyword_boost if plan else 1.0),
    )
    capped = merged[:max_iter]

    ctx.add_event(
        "retrieval_iter",
        iter=state.get("iteration", 0) + 1,
        bm25_retrieved=len(bm25_hits),
        filter_retrieved=len(filt_hits),
        candidates_in=len(merged),
        candidates_after_cap=len(capped),
        candidate_ids=[hit["company_id"] for hit in capped],
        top_score=capped[0]["score"] if capped else 0.0,
        use_bm25=use_bm25,
        use_filters=use_filters,
        mandatory_filters_applied=hard_filters,
        eligible_count=eligible_count,
    )
    return {
        "candidates": capped,
        "iteration": state.get("iteration", 0) + 1,
    }


# ─── revise_search ─────────────────────────────────────────────────────
async def revise_search_node(state: AgentState, ctx: RunContext, llm: LLMClient) -> dict:
    """One-shot keyword revision preserving all original requirements."""
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
            max_attempts=1,
            stage="revise_search",
        )
        proposed = sanitize_mandate(new_raw, raw_query=state.get("raw_query", ""))
        # Revision may broaden lexical recall, but it cannot silently relax
        # mandatory filters or must-haves from the original user mandate.
        revised_keywords = proposed.filters.keywords or mandate.filters.keywords
        new_filters = mandate.filters.model_copy(
            update={"keywords": list(dict.fromkeys(revised_keywords))}
        )
        new_mandate = mandate.model_copy(update={"filters": new_filters})
        ok = True
    except _LLM_RECOVERABLE_ERRORS as exc:
        logger.warning("revise_search.failed", error=str(exc))
        ok = False
        new_mandate = mandate
        ctx.add_error("revise_search", exc)

    ctx.add_event(
        "revise_search",
        ok=ok,
        duration_ms=int((time.perf_counter() - t0) * 1000),
        iteration=state.get("iteration", 0),
        new_keywords=new_mandate.filters.keywords if ok else None,
        mandatory_filters_preserved=True,
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
    """Validate at most ten eligible candidates, with grounded evidence."""
    limits = _limits(state)
    validation_limit = min(
        10,
        max_validate,
        int(limits.get("max_candidates_to_validate", 10)),
    )
    cands = (state.get("candidates") or [])[:validation_limit]
    if not cands:
        ctx.add_event(
            "validation",
            ok=True,
            outcome="no_candidates",
            candidates_in=0,
            validated=0,
            kept=0,
            dropped=0,
        )
        return {"validation": [], "validation_ok": True}

    # Search synonyms are retrieval hints, never new acceptance criteria.
    mandate = state.get("original_mandate") or state.get("mandate")
    if mandate is None:
        return {"validation": [], "validation_ok": False}

    records = await company_repo.fetch_by_ids([h["company_id"] for h in cands])
    by_id: dict[int, CompanyRecord] = {record.id: record for record in records}
    policy = EligibilityPolicy(mandate.filters)
    eligible: list[tuple[ScoredHit, CompanyRecord]] = [
        (hit, by_id[hit["company_id"]])
        for hit in cands
        if hit["company_id"] in by_id and policy.matches(by_id[hit["company_id"]])
    ]
    eligibility_dropped = len(cands) - len(eligible)
    criteria = semantic_criteria(mandate)
    semantic_validation_needed = bool(criteria)

    if not eligible:
        ctx.add_event(
            "validation",
            ok=True,
            outcome="no_eligible_candidates",
            candidates_in=len(cands),
            eligibility_dropped=eligibility_dropped,
            validated=0,
            kept=0,
            dropped=len(cands),
        )
        return {"validation": [], "validation_ok": True}

    if not semantic_validation_needed:
        deterministic: list[ValidatedItem] = []
        for hit, record in eligible:
            evidence = policy.evidence(record)
            deterministic.append(
                {
                    "company": record,
                    "score": hit["score"],
                    "relevant": bool(evidence),
                    "evidence": evidence,
                    "reason": "mandatory structured filters verified deterministically",
                }
            )
        ctx.add_event(
            "validation",
            ok=True,
            outcome="deterministic",
            candidates_in=len(cands),
            eligibility_dropped=eligibility_dropped,
            validated=len(deterministic),
            kept=sum(1 for item in deterministic if item["relevant"]),
            dropped=eligibility_dropped,
        )
        return {"validation": deterministic, "validation_ok": True}

    t0 = time.perf_counter()
    decisions, ok = await SemanticValidationService(llm).validate(
        state.get("raw_query", ""), [record for _, record in eligible], criteria, ctx,
    )
    validated: list[ValidatedItem] = []
    for hit, rec in eligible:
        decision = decisions[rec.id]
        evidence = policy.evidence(rec) + decision.evidence if decision.relevant else []
        validated.append(
            {
                "company": rec,
                "score": hit["score"],
                "relevant": decision.relevant,
                "evidence": evidence,
                "reason": decision.reason,
                "criteria": decision.criteria,
            }
        )

    ctx.add_event(
        "validation",
        ok=ok,
        duration_ms=int((time.perf_counter() - t0) * 1000),
        outcome="criterion_validation" if ok else "unverified",
        criteria=[criterion.model_dump() for criterion in criteria],
        decisions=[
            {"company_id": item["company"].id, "relevant": item["relevant"],
             "criteria": [assessment.model_dump() for assessment in item["criteria"]]}
            for item in validated
        ],
        candidates_in=len(cands),
        eligibility_dropped=eligibility_dropped,
        validated=len(validated),
        kept=sum(1 for v in validated if v["relevant"]),
        dropped=eligibility_dropped + sum(1 for v in validated if not v["relevant"]),
    )
    return {"validation": validated, "validation_ok": ok}


# ─── finalize ──────────────────────────────────────────────────────────
async def finalize_node(state: AgentState, ctx: RunContext, max_final: int) -> dict:
    """Sort by score, drop not-relevant, cap at max_final."""
    validation = state.get("validation") or []
    relevant = [
        item
        for item in validation
        if item.get("relevant") and item.get("evidence")
    ]
    final = sorted(
        relevant,
        key=lambda item: (-item.get("score", 0.0), item["company"].id),
    )[: min(10, max_final)]
    ctx.add_event(
        "finalize",
        validation_count=len(validation),
        relevant_count=len(relevant),
        final_count=len(final),
    )
    return {"final": final, "latency_ms": ctx.elapsed_ms()}
