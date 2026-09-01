"""Pydantic DTOs for the public /search API surface."""
from __future__ import annotations

from pydantic import BaseModel, Field

from comparables.schemas.company import CompanyRecord
from comparables.schemas.mandate import ParsedMandate


class Evidence(BaseModel):
    """Grounded support for a candidate: a literal substring of the record."""

    field: str = Field(description="Which field of the record this span comes from.")
    span: str = Field(min_length=1, description="Exact substring of `field`.")


class Candidate(BaseModel):
    """One returned company with its grounded evidence and a 0..1 score."""

    company: CompanyRecord
    score: float = Field(ge=0.0, le=1.0)
    relevant: bool = True
    evidence: list[Evidence] = Field(default_factory=list)
    reason: str = ""


class SearchRequest(BaseModel):
    """Public request body for POST /api/v1/search."""

    query: str = Field(min_length=1, max_length=1000)


class SearchResponse(BaseModel):
    """Public response: the parsed mandate + final list of grounded candidates."""

    run_id: str
    query: str
    mandate: ParsedMandate | None = None
    final: list[Candidate] = Field(default_factory=list)
    llm_calls: int = 0
    revised_search: bool = False
    iterations: int = 0
    latency_ms: int = 0
    errors: list[dict] = Field(default_factory=list)
