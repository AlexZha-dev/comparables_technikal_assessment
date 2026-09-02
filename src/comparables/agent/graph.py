"""Build the LangGraph StateGraph that orchestrates the agent."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from comparables.agent.edges import (
    after_parse,
    after_plan,
    after_retrieve,
    should_revise_again,
)
from comparables.agent.nodes import (
    finalize_node,
    parse_mandate_node,
    plan_search_node,
    retrieve_and_score_node,
    revise_search_node,
    validate_candidates_node,
)
from comparables.agent.state import AgentState
from comparables.core.context import RunContext
from comparables.llm.client import LLMClient
from comparables.tools.registry import ToolRegistry


def build_graph(
    *,
    ctx: RunContext,
    llm: LLMClient,
    registry: ToolRegistry,
    company_repo,
    limits: dict[str, int],
    w_bm: float,
    w_f: float,
    max_iterations: int,
    max_revised: int,
    max_validate: int,
    max_final: int,
):
    """Compile a StateGraph. The same builder is reused for every request."""

    # Wrap nodes to inject deps (LangGraph only passes state).
    async def _parse(state):
        return await parse_mandate_node(state, ctx, llm)

    async def _plan(state):
        return await plan_search_node(state, ctx, llm)

    async def _retrieve(state):
        return await retrieve_and_score_node(state, ctx, registry, company_repo, w_bm, w_f)

    async def _revise(state):
        return await revise_search_node(state, ctx, llm)

    async def _validate(state):
        return await validate_candidates_node(state, ctx, llm, company_repo, max_validate)

    async def _finalize(state):
        return await finalize_node(state, ctx, max_final)

    def _after_retrieve(state):
        return after_retrieve(
            state, max_iterations=max_iterations, max_revised=max_revised
        )

    g = StateGraph(AgentState)
    g.add_node("parse_mandate", _parse)
    g.add_node("plan_search", _plan)
    g.add_node("retrieve_and_score", _retrieve)
    g.add_node("revise_search", _revise)
    g.add_node("validate_candidates", _validate)
    g.add_node("finalize", _finalize)

    g.add_edge(START, "parse_mandate")
    g.add_conditional_edges(
        "parse_mandate", after_parse, {"plan_search": "plan_search", "finalize": "finalize"}
    )
    g.add_conditional_edges(
        "plan_search",
        after_plan,
        {
            "retrieve_and_score": "retrieve_and_score",
            "validate_candidates": "validate_candidates",
        },
    )
    g.add_conditional_edges(
        "retrieve_and_score",
        _after_retrieve,
        {"revise_search": "revise_search", "validate_candidates": "validate_candidates"},
    )
    g.add_conditional_edges(
        "revise_search",
        should_revise_again,
        {"retrieve_and_score": "retrieve_and_score", "validate_candidates": "validate_candidates"},
    )
    g.add_edge("validate_candidates", "finalize")
    g.add_edge("finalize", END)

    return g.compile()


__all__ = ["build_graph"]
