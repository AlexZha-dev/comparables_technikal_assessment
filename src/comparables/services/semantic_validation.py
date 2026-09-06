"""One bounded LLM batch with criterion-level, fail-closed decisions."""

from __future__ import annotations

import json
from typing import Protocol

from pydantic import BaseModel, ValidationError

from comparables.agent.prompts import VALIDATE_CANDIDATE_SYSTEM
from comparables.agent.semantic import SemanticAcceptancePolicy, SemanticDecision
from comparables.core.context import RunContext
from comparables.core.exceptions import (
    BudgetExceededError,
    LLMSchemaError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from comparables.schemas.company import CompanyRecord
from comparables.schemas.search import BatchVerdicts, SemanticCriterion


class StructuredCompleter(Protocol):
    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema_model: type[BaseModel],
        ctx: RunContext | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_attempts: int | None = None,
        stage: str = "complete_json",
    ) -> BaseModel: ...


class SemanticValidationService:
    def __init__(self, llm: StructuredCompleter) -> None:
        self._llm = llm
        self._policy = SemanticAcceptancePolicy()

    async def validate(
        self,
        query: str,
        records: list[CompanyRecord],
        criteria: list[SemanticCriterion],
        ctx: RunContext,
    ) -> tuple[dict[int, SemanticDecision], bool]:
        if len(records) > 10:
            raise ValueError("At most 10 records may enter semantic validation")
        if not records:
            return {}, True
        ok = False
        batch = BatchVerdicts(verdicts=[])
        try:
            if not criteria or len(criteria) > 12:
                raise LLMSchemaError("Expected between 1 and 12 atomic semantic requirements")
            payload = {
                "query": query,
                "criteria": [criterion.model_dump() for criterion in criteria],
                "companies": [record.model_dump() for record in records],
            }
            response = await self._llm.complete_json(
                system=VALIDATE_CANDIDATE_SYSTEM,
                user=json.dumps(payload, ensure_ascii=False),
                schema_model=BatchVerdicts,
                ctx=ctx,
                temperature=0.0,
                max_tokens=min(8192, 512 + 128 * len(records) * len(criteria)),
                max_attempts=2,
                stage="validate_candidates",
            )
            # Validate even injected/custom adapters at the boundary.
            batch = BatchVerdicts.model_validate(response.model_dump())
            ids = [verdict.company_id for verdict in batch.verdicts]
            expected = {record.id for record in records}
            if set(ids) != expected or len(ids) != len(expected):
                raise LLMSchemaError(
                    "Validator must return exactly one verdict for each supplied record"
                )
            expected_criteria = {criterion.criterion_id for criterion in criteria}
            if any(
                {item.criterion_id for item in verdict.criteria} != expected_criteria
                or len(verdict.criteria) != len(expected_criteria)
                for verdict in batch.verdicts
            ):
                raise LLMSchemaError("Validator omitted, duplicated, or invented a criterion")
            ok = True
        except (
            LLMSchemaError,
            LLMUnavailableError,
            LLMTimeoutError,
            BudgetExceededError,
            ValidationError,
        ) as exc:
            ctx.add_error("validate_candidates", exc)
        by_id = {verdict.company_id: verdict for verdict in batch.verdicts} if ok else {}
        return {
            record.id: self._policy.assess(record, criteria, by_id.get(record.id))
            for record in records
        }, ok
