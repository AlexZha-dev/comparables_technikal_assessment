"""Tests for LLM structured helpers."""
from __future__ import annotations

from comparables.llm.structured import extract_json_object, parse_strict, schema_instructions
from comparables.schemas.mandate import FilterSpec, ParsedMandate


def test_schema_instructions_mentions_required():
    out = schema_instructions(ParsedMandate)
    assert "Required keys" in out
    assert "intent" in out
    assert "filters" in out
    assert "industries" in out


def test_extract_json_object_plain():
    text = '{"a": 1, "b": [2, 3]}'
    assert extract_json_object(text) == text


def test_extract_json_object_fenced():
    text = 'Here:\n```json\n{"a": 1}\n```\nbye'
    out = extract_json_object(text)
    assert out == '{"a": 1}'


def test_extract_json_object_embedded():
    text = 'prose {"k": "v"} more prose'
    assert extract_json_object(text) == '{"k": "v"}'


def test_parse_strict_valid_mandate():
    raw = '{"intent": "find fintech", "filters": {"industries": ["Fintech"]}, "must_haves": []}'
    m = parse_strict(raw, ParsedMandate)
    assert m.intent == "find fintech"
    assert m.filters.industries == ["Fintech"]


def test_parse_strict_invalid_raises():
    import pytest
    from comparables.core.exceptions import LLMSchemaError

    bad = '{"intent": 123}'  # intent must be str
    with pytest.raises(LLMSchemaError):
        parse_strict(bad, ParsedMandate)
