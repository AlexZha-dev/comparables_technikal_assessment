"""Domain exceptions. Mapped to HTTP responses by exception handlers in main.py."""
from __future__ import annotations


class ComparablesError(Exception):
    """Base for all domain errors."""

    run_id: str | None = None


# ─── Data / index errors ───────────────────────────────────────────────
class IndexNotFoundError(ComparablesError):
    """SQLite or BM25 pickle missing — index-dataset skill must run first."""


class CompanyNotFoundError(ComparablesError):
    """Company id not present in the catalog."""


# ─── Mandate / parse errors ────────────────────────────────────────────
class MandateParseError(ComparablesError):
    """LLM failed to produce a valid mandate after retries."""


# ─── LLM / provider errors ─────────────────────────────────────────────
class LLMUnavailableError(ComparablesError):
    """Ollama / OpenAI-compat endpoint not reachable."""


class LLMTimeoutError(ComparablesError):
    """LLM call exceeded the configured timeout."""


class LLMSchemaError(ComparablesError):
    """LLM returned a payload that doesn't match the Pydantic schema."""


# ─── Workflow / budget errors ──────────────────────────────────────────
class BudgetExceededError(ComparablesError):
    """Hit a hard limit (iterations, LLM calls, candidates)."""


class WorkflowTimeoutError(ComparablesError):
    """LangGraph run exceeded settings.workflow_timeout_s."""


# ─── Retrieval errors ──────────────────────────────────────────────────
class RetrievalError(ComparablesError):
    """Generic retrieval failure."""
