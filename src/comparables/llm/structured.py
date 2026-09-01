"""Structured-output helpers: prompt construction, JSON parsing, schema validation.

Ollama's OpenAI-compat endpoint does NOT support `response_format=json_schema`
as a server-side constraint. We embed the schema in the system prompt and parse
the response content as JSON. Pydantic does the rest.
"""
from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from comparables.core.exceptions import LLMSchemaError, LLMTimeoutError, LLMUnavailableError

T = TypeVar("T", bound=BaseModel)


def schema_for(model: type[BaseModel]) -> dict:
    """JSON schema for a Pydantic model, suitable for embedding in a prompt."""
    return model.model_json_schema()


def schema_instructions(model: type[BaseModel]) -> str:
    """Render a Pydantic schema as a tight set of instructions for the LLM."""
    schema = schema_for(model)
    required = schema.get("required", [])
    props = schema.get("properties", {})
    lines = [
        "Return ONLY a JSON object matching this schema. No prose, no markdown fences.",
        "Do not include keys not listed in `properties`.",
        f"Required keys: {sorted(required)}",
        "",
        "Schema (JSON Schema draft-07-ish):",
        json.dumps(schema, ensure_ascii=False),
    ]
    return "\n".join(lines)


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str) -> str:
    """Best-effort: find the first {...} block in `text`.

    Handles models that wrap JSON in prose or fences.
    """
    text = text.strip()
    if text.startswith("```"):
        # strip ```json ... ```
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[:-3]
    m = _JSON_OBJECT_RE.search(text)
    if not m:
        raise LLMSchemaError(f"No JSON object found in LLM response: {text[:200]!r}")
    return m.group(0)


def parse_strict(text: str, model: type[T]) -> T:
    """Parse `text` as JSON and validate against `model`. Raises LLMSchemaError."""
    try:
        raw = extract_json_object(text)
        obj = json.loads(raw)
    except (json.JSONDecodeError, LLMSchemaError) as exc:
        raise LLMSchemaError(f"JSON parse failed: {exc}") from exc
    try:
        return model.model_validate(obj)
    except ValidationError as exc:
        raise LLMSchemaError(f"Schema validation failed: {exc}") from exc


__all__ = [
    "schema_for",
    "schema_instructions",
    "extract_json_object",
    "parse_strict",
    "LLMSchemaError",
    "LLMTimeoutError",
    "LLMUnavailableError",
]
