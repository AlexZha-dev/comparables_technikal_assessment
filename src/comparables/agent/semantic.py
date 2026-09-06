"""Pure requirement compilation and fail-closed semantic acceptance policy."""

from __future__ import annotations

from dataclasses import dataclass

from comparables.schemas.company import CompanyRecord
from comparables.schemas.mandate import ParsedMandate
from comparables.schemas.search import (
    CandidateVerdict,
    CriterionAssessment,
    Evidence,
    SemanticCriterion,
)


def retrieval_query(mandate: ParsedMandate) -> str:
    """Semantic claims remain searchable even if a parser omits keyword hints."""
    terms = mandate.filters.keywords or [
        criterion.requirement for criterion in semantic_criteria(mandate)
    ]
    return " ".join(term.strip() for term in terms if term.strip())


def semantic_criteria(mandate: ParsedMandate) -> list[SemanticCriterion]:
    """Prefer explicit business claims; support legacy keyword-only mandates.

    Retrieval revisions never supply this mandate. Alias folding only avoids
    duplicate legacy checks; it is NOT an evidence or acceptance heuristic.
    """
    claims = [*mandate.semantic_requirements, *mandate.must_haves] or mandate.filters.keywords
    unique: dict[str, str] = {}
    for claim in claims:
        claim = claim.strip()
        key = " ".join(claim.casefold().split())
        if key in {"ai", "ai-driven", "ai-powered", "artificial intelligence"}:
            key = "ai"
        if key:
            unique.setdefault(key, claim)
    return [
        SemanticCriterion(criterion_id=f"c{index}", requirement=claim)
        for index, claim in enumerate(unique.values(), 1)
    ]


@dataclass(frozen=True)
class SemanticDecision:
    relevant: bool
    evidence: list[Evidence]
    criteria: list[CriterionAssessment]
    reason: str


class SemanticAcceptancePolicy:
    """Every requested claim must be supported with traceable field evidence.

    This enforces the model contract, not the truth of the model's semantic
    judgment. Relevance labels evaluate that remaining source of error.
    """

    def assess(
        self,
        record: CompanyRecord,
        criteria: list[SemanticCriterion],
        verdict: CandidateVerdict | None,
    ) -> SemanticDecision:
        expected = {criterion.criterion_id for criterion in criteria}
        supplied = [item.criterion_id for item in verdict.criteria] if verdict else []
        complete = bool(
            verdict
            and verdict.company_id == record.id
            and set(supplied) == expected
            and len(supplied) == len(expected)
        )
        by_id = (
            {item.criterion_id: item for item in verdict.criteria} if complete and verdict else {}
        )
        assessments: list[CriterionAssessment] = []
        for criterion in criteria:
            item = by_id.get(criterion.criterion_id)
            traceable = (
                item is not None
                and bool(item.evidence)
                and all(ev.span in getattr(record, ev.field) for ev in item.evidence)
            )
            status = item.status if item and traceable else "insufficient_evidence"
            assessments.append(
                CriterionAssessment(
                    **criterion.model_dump(),
                    status=status,
                    evidence=item.evidence if item and traceable else [],
                )
            )
        relevant = (
            bool(assessments)
            and complete
            and all(item.status == "supported" for item in assessments)
        )
        evidence = (
            [Evidence(**ev.model_dump()) for item in assessments for ev in item.evidence]
            if relevant
            else []
        )
        return SemanticDecision(
            relevant=relevant,
            evidence=evidence,
            criteria=assessments,
            reason="every semantic requirement supported"
            if relevant
            else "one or more requirements contradicted, unverified, or missing",
        )
