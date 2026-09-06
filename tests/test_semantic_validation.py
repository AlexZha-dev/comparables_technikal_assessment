"""Contract tests, not a substitute for live-model semantic evaluation."""

import json

import pytest

from comparables.agent.semantic import SemanticAcceptancePolicy, semantic_criteria
from comparables.core.context import RunContext
from comparables.core.exceptions import LLMUnavailableError
from comparables.schemas.company import CompanyRecord
from comparables.schemas.mandate import FilterSpec, ParsedMandate
from comparables.schemas.search import (
    BatchVerdicts,
    CandidateVerdict,
    CriterionVerdict,
    SemanticCriterion,
    SemanticEvidence,
)
from comparables.services.semantic_validation import SemanticValidationService


def record():
    return CompanyRecord(
        id=1,
        name="Mobility",
        description="Develops autonomous driving software. Does not use machine learning.",
        industry="Automotive",
        location="Netherlands",
        founded_year=2020,
        employee_count=200,
        revenue_range="10M-50M",
    )


def requirements():
    return [
        SemanticCriterion(criterion_id="c1", requirement="Develops autonomous driving software"),
        SemanticCriterion(criterion_id="c2", requirement="Uses machine learning"),
    ]


@pytest.mark.parametrize("second_status", ["contradicted", "insufficient_evidence"])
def test_all_mandatory_claims_must_be_supported(second_status):
    verdict = CandidateVerdict(
        company_id=1,
        criteria=[
            CriterionVerdict(
                criterion_id="c1",
                status="supported",
                evidence=[
                    SemanticEvidence(
                        field="description", span="Develops autonomous driving software."
                    )
                ],
            ),
            CriterionVerdict(
                criterion_id="c2",
                status=second_status,
                evidence=[
                    SemanticEvidence(field="description", span="Does not use machine learning.")
                ]
                if second_status == "contradicted"
                else [],
            ),
        ],
    )
    decision = SemanticAcceptancePolicy().assess(record(), requirements(), verdict)
    assert not decision.relevant
    assert not decision.evidence
    assert decision.criteria[1].status == second_status


@pytest.mark.parametrize("bad_span", ["Invented evidence", "Uses machine learning"])
def test_traceable_evidence_is_required_per_criterion(bad_span):
    verdict = CandidateVerdict(
        company_id=1,
        criteria=[
            CriterionVerdict(
                criterion_id=c.criterion_id,
                status="supported",
                evidence=[SemanticEvidence(field="description", span=bad_span)],
            )
            for c in requirements()
        ],
    )
    assert not SemanticAcceptancePolicy().assess(record(), requirements(), verdict).relevant


def test_retrieval_keywords_do_not_replace_explicit_business_requirements():
    mandate = ParsedMandate(
        intent="test",
        filters=FilterSpec(keywords=["AI", "cloud", "platform"]),
        semantic_requirements=["Uses AI in its own product"],
    )
    assert [criterion.requirement for criterion in semantic_criteria(mandate)] == [
        "Uses AI in its own product"
    ]


@pytest.mark.asyncio
async def test_provider_failure_never_accepts_literal_keyword_match():
    class Down:
        async def complete_json(self, **kwargs):
            kwargs["ctx"].begin_llm_call()
            raise LLMUnavailableError("offline")

    ctx = RunContext.new()
    decisions, ok = await SemanticValidationService(Down()).validate(
        "machine learning", [record()], requirements(), ctx
    )
    assert not ok and not decisions[1].relevant
    assert ctx.llm_calls == 1
    assert ctx.errors


@pytest.mark.parametrize("invalid_ids", [["c1"], ["c1", "c1"], ["c1", "invented"]])
@pytest.mark.asyncio
async def test_missing_duplicate_or_invented_criteria_fail_closed(invalid_ids):
    class Incomplete:
        async def complete_json(self, **kwargs):
            payload = json.loads(kwargs["user"])
            assert len(payload["companies"]) == 1 and len(payload["criteria"]) == 2
            return BatchVerdicts(
                verdicts=[
                    CandidateVerdict(
                        company_id=1,
                        criteria=[
                            CriterionVerdict(
                                criterion_id=cid,
                                status="supported",
                                evidence=[
                                    SemanticEvidence(field="description", span=record().description)
                                ],
                            )
                            for cid in invalid_ids
                        ],
                    )
                ]
            )

    decisions, ok = await SemanticValidationService(Incomplete()).validate(
        "query", [record()], requirements(), RunContext.new()
    )
    assert not ok and not decisions[1].relevant


@pytest.mark.asyncio
async def test_more_than_ten_records_rejected_before_provider_call():
    class MustNotCall:
        async def complete_json(self, **kwargs):
            pytest.fail("provider called before input cap")

    with pytest.raises(ValueError, match="At most 10"):
        await SemanticValidationService(MustNotCall()).validate(
            "query", [record()] * 11, requirements(), RunContext.new()
        )


@pytest.mark.parametrize("company_ids", [[], [1, 1], [999]])
@pytest.mark.asyncio
async def test_missing_duplicate_or_invented_company_fails_closed(company_ids):
    class BadIds:
        async def complete_json(self, **kwargs):
            return BatchVerdicts(
                verdicts=[
                    CandidateVerdict(
                        company_id=cid,
                        criteria=[
                            CriterionVerdict(
                                criterion_id=c.criterion_id, status="insufficient_evidence"
                            )
                            for c in requirements()
                        ],
                    )
                    for cid in company_ids
                ]
            )

    decisions, ok = await SemanticValidationService(BadIds()).validate(
        "query", [record()], requirements(), RunContext.new()
    )
    assert not ok and not decisions[1].relevant


def test_accepts_only_complete_supported_traceable_verdict():
    company = record().model_copy(
        update={"description": "Develops autonomous driving software using machine learning."}
    )
    verdict = CandidateVerdict(
        company_id=1,
        criteria=[
            CriterionVerdict(
                criterion_id=c.criterion_id,
                status="supported",
                evidence=[SemanticEvidence(field="description", span=company.description)],
            )
            for c in requirements()
        ],
    )
    decision = SemanticAcceptancePolicy().assess(company, requirements(), verdict)
    assert decision.relevant and len(decision.criteria) == 2
    assert all(item.status == "supported" for item in decision.criteria)
