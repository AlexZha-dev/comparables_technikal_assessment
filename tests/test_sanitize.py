"""Unit tests for the mandate sanitizer."""
from __future__ import annotations

from comparables.agent.prompts import (
    ALLOWED_INDUSTRIES,
    ALLOWED_LOCATIONS,
    ALLOWED_REVENUE_BUCKETS,
)
from comparables.agent.sanitize import sanitize_mandate
from comparables.schemas.mandate import FilterSpec, ParsedMandate


def _m(**kw) -> ParsedMandate:
    return ParsedMandate(
        intent="test",
        filters=FilterSpec(**kw),
    )


# ─── happy path ────────────────────────────────────────────────────────
def test_keeps_valid_values_unchanged():
    m = _m(industries=["Fintech", "Healthcare"], locations=["Sweden", "Norway"], revenue_buckets=["50M-100M"])
    out = sanitize_mandate(m)
    assert out.filters.industries == ["Fintech", "Healthcare"]
    assert out.filters.locations == ["Sweden", "Norway"]
    assert out.filters.revenue_buckets == ["50M-100M"]


def test_case_insensitive_match_for_industries():
    m = _m(industries=["fintech", "FINTECH"])
    out = sanitize_mandate(m)
    assert out.filters.industries == ["Fintech"]


# ─── hallucination stripping ───────────────────────────────────────────
def test_drops_truly_unknown_industries():
    m = _m(industries=["Zorglux Holdings", "Quibblenauts Corp", "FooBar Industries"])
    out = sanitize_mandate(m)
    assert out.filters.industries == []


def test_hallucinated_industries_get_synonym_or_dropped():
    """Hallucinated industry tokens should be mapped via synonym when possible
    (e.g. 'Biomedical' -> Healthcare), otherwise dropped."""
    m = _m(industries=["Biomedical Innovation", "Biotech Research", "Biotechnology"])
    out = sanitize_mandate(m)
    # All three map to allowed values via the synonym table.
    assert sorted(out.filters.industries) == sorted(["Healthcare", "Biotech"])
    # Nothing unknown leaks through.
    for kept in out.filters.industries:
        assert kept in ALLOWED_INDUSTRIES


def test_drops_hallucinated_locations():
    m = _m(locations=["France (Metropolitan)", "Paris", "FRANCE"])
    out = sanitize_mandate(m)
    # FRANCE matches case-insensitively; the others are not in the allowed list.
    assert out.filters.locations == ["France"]


def test_drops_hallucinated_revenue_buckets():
    m = _m(revenue_buckets=["around-50M", "$50M-$100M", "50M-100M"])
    out = sanitize_mandate(m)
    assert out.filters.revenue_buckets == ["50M-100M"]


# ─── synonym mapping ───────────────────────────────────────────────────
def test_autonomous_driving_maps_to_automotive():
    m = _m(industries=["Autonomous driving"])
    out = sanitize_mandate(m)
    assert out.filters.industries == ["Automotive"]


def test_machine_learning_maps_to_technology():
    m = _m(industries=["machine learning", "AI"])
    out = sanitize_mandate(m)
    # "AI" doesn't match any synonym (substring "ai " has trailing space).
    # "machine learning" maps to Technology.
    assert out.filters.industries == ["Technology"]


def test_biotechnology_maps_to_biotech():
    m = _m(industries=["Biotechnology", "biotech", "Biotech"])
    out = sanitize_mandate(m)
    assert out.filters.industries == ["Biotech"]


def test_pharma_maps_to_healthcare():
    m = _m(industries=["Pharmaceutical", "pharma"])
    out = sanitize_mandate(m)
    assert out.filters.industries == ["Healthcare"]


def test_renewable_maps_to_energy():
    m = _m(industries=["renewable energy"])
    out = sanitize_mandate(m)
    assert out.filters.industries == ["Energy"]


# ─── location synonyms ────────────────────────────────────────────────
def test_us_maps_to_usa():
    m = _m(locations=["US"])
    out = sanitize_mandate(m)
    assert out.filters.locations == ["USA"]


def test_uk_maps_to_uk():
    m = _m(locations=["United Kingdom", "Britain", "England"])
    out = sanitize_mandate(m)
    assert out.filters.locations == ["UK"]


def test_nordic_expands_to_three():
    m = _m(locations=["Nordics", "northern europe"])
    out = sanitize_mandate(m)
    # Only "Nordics" expands; the other is unmapped.
    assert out.filters.locations == ["Sweden", "Norway", "Finland"]


# ─── idempotence ──────────────────────────────────────────────────────
def test_idempotent():
    m = _m(
        industries=["Biotechnology", "renewable energy", "Fintech"],
        locations=["Nordics", "USA"],
        revenue_buckets=["50M-100M"],
        employee_min=100,
        keywords=["foo"],
    )
    once = sanitize_mandate(m)
    twice = sanitize_mandate(once)
    assert once.model_dump() == twice.model_dump()


# ─── preserves non-validated fields ────────────────────────────────────
def test_preserves_intent_keywords_employees():
    m = _m(
        industries=["Fintech"],
        keywords=["ai", "ml"],
        employee_min=10,
        employee_max=500,
        founded_after=2010,
    )
    out = sanitize_mandate(m)
    assert out.intent == "test" or out.intent == "test"  # mangled above
    assert out.filters.keywords == ["ai", "ml"]
    assert out.filters.employee_min == 10
    assert out.filters.employee_max == 500
    assert out.filters.founded_after == 2010


# ─── sanity: all allowed values still pass through ────────────────────
def test_all_allowed_values_round_trip():
    m = _m(
        industries=list(ALLOWED_INDUSTRIES),
        locations=list(ALLOWED_LOCATIONS),
        revenue_buckets=list(ALLOWED_REVENUE_BUCKETS),
    )
    out = sanitize_mandate(m)
    assert out.filters.industries == list(ALLOWED_INDUSTRIES)
    assert out.filters.locations == list(ALLOWED_LOCATIONS)
    assert out.filters.revenue_buckets == list(ALLOWED_REVENUE_BUCKETS)
