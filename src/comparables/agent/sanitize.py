"""Post-process parsed mandate to drop values that don't exist in the dataset.

The LLM is instructed to use only allowed values, but smaller models occasionally
hallucinate near-misses ("Biotechnology", "France (Metropolitan)"). Those values
then become useless `WHERE x IN (...)` filters. This module sanitizes the mandate
in-place:

1. Strips any industry/location/revenue value not in the allowed set.
2. Maps a small set of user-language synonyms (case-insensitive substring) to
   canonical allowed values BEFORE the membership check.

Returns a NEW ParsedMandate; the input is not mutated.
"""
from __future__ import annotations

import re

from comparables.agent.prompts import (
    ALLOWED_INDUSTRIES,
    ALLOWED_LOCATIONS,
    ALLOWED_REVENUE_BUCKETS,
)
from comparables.schemas.mandate import FilterSpec, ParsedMandate

# Synonyms that map cleanly to a canonical allowed value. Order matters:
# first hit wins. Keys are case-insensitive substrings of the user-issued token.
_INDUSTRY_SYNONYMS: list[tuple[str, str]] = [
    ("biotech", "Biotech"),
    ("biotechn", "Biotech"),  # Biotechnology -> Biotech
    ("biomedic", "Healthcare"),  # Biomedical -> Healthcare
    ("pharma", "Healthcare"),
    ("pharmaceutical", "Healthcare"),
    ("autonomous driving", "Automotive"),
    ("autonomous vehicle", "Automotive"),
    ("self-driving", "Automotive"),
    ("fintech", "Fintech"),
    ("financial tech", "Fintech"),
    ("financial technology", "Fintech"),
    ("ai-driven", "Technology"),
    ("ai ", "Technology"),
    ("artificial intelligence", "Technology"),
    ("machine learning", "Technology"),
    ("deep tech", "Technology"),
    ("robotics", "Automotive"),
    ("renewable", "Energy"),
    ("clean energy", "Energy"),
    ("solar", "Energy"),
    ("wind", "Energy"),
    ("edtech", "Education"),
    ("healthtech", "Healthcare"),
    ("medtech", "Healthcare"),
    ("medical", "Healthcare"),
    ("logistics", "Logistics"),
    ("supply chain", "Logistics"),
    ("transport", "Logistics"),
    ("ecommerce", "Retail"),
    ("e-commerce", "Retail"),
    ("telecom", "Telecom"),
    ("telecommunications", "Telecom"),
    ("5g", "Telecom"),
]

_LOCATION_SYNONYMS: list[tuple[str, str | None]] = [
    ("nordic", None),  # special: expands, not a single-value alias
    ("scandinav", None),
    ("us", "USA"),
    ("united states", "USA"),
    ("u.s.", "USA"),
    ("u.k.", "UK"),
    ("united kingdom", "UK"),
    ("great britain", "UK"),
    ("britain", "UK"),
    ("england", "UK"),
    ("holland", "Netherlands"),
    ("the netherlands", "Netherlands"),
]

# Multi-word synonym expansions (substring -> list of allowed values)
_NORDIC_LOCATIONS = ["Sweden", "Norway", "Finland"]

_EXPLICIT_INDUSTRY_RULES: list[tuple[tuple[str, ...], str]] = [
    (("autonomous driving", "autonomous vehicle", "self-driving"), "Automotive"),
    (("biotech", "biotechnology"), "Biotech"),
    (("fintech", "financial technology"), "Fintech"),
    (("renewable energy", "clean energy", "energy compan"), "Energy"),
    (("healthcare", "health care"), "Healthcare"),
    (("automotive",), "Automotive"),
    (("education", "edtech"), "Education"),
    (("logistics", "supply chain"), "Logistics"),
    (("retail", "e-commerce", "ecommerce"), "Retail"),
    (("telecom", "telecommunications"), "Telecom"),
    (("technology compan",), "Technology"),
]

_KEYWORD_PHRASES = (
    "artificial intelligence",
    "machine learning",
    "autonomous driving",
    "self-driving",
    "renewable energy",
    "fraud detection",
    "enterprise",
    "b2b",
)


def _normalize_ind(token: str) -> str | None:
    t = token.strip().lower()
    if not t:
        return None
    # First, allow exact (case-insensitive) match against the allowed set.
    for allowed in ALLOWED_INDUSTRIES:
        if t == allowed.lower():
            return allowed
    # Then, synonym map.
    for substr, canonical in _INDUSTRY_SYNONYMS:
        if substr in t:
            # Validate the canonical against the allowed list to avoid stale synonyms.
            if canonical in ALLOWED_INDUSTRIES:
                return canonical
            return None
    return None


def _normalize_loc(token: str) -> list[str] | None:
    """Returns a list of allowed locations (possibly more than one — e.g. Nordics),
    or None if the token cannot be mapped."""
    t = token.strip().lower()
    if not t:
        return None
    # Exact case-insensitive match.
    for allowed in ALLOWED_LOCATIONS:
        if t == allowed.lower():
            return [allowed]
    # Synonym map. None means "expand" — see below.
    for substr, canonical in _LOCATION_SYNONYMS:
        if substr == "us" and not re.search(r"\bus\b", t):
            continue
        if substr in t:
            if canonical is None:
                # Expansions are handled by substring triggers — only used
                # for 'nordic'/'scandinav' which we expand below.
                if substr in ("nordic", "scandinav"):
                    return list(_NORDIC_LOCATIONS)
                return None
            if canonical in ALLOWED_LOCATIONS:
                return [canonical]
            return None
    return None


def _normalize_rev(token: str) -> str | None:
    t = token.strip()
    if not t:
        return None
    for allowed in ALLOWED_REVENUE_BUCKETS:
        if t == allowed:
            return allowed
    return None


def _explicit_industries(raw_query: str) -> list[str]:
    text = raw_query.casefold()
    return [
        industry
        for needles, industry in _EXPLICIT_INDUSTRY_RULES
        if any(needle in text for needle in needles)
    ]


def _explicit_locations(raw_query: str) -> list[str]:
    text = raw_query.casefold()
    if "nordic" in text or "scandinav" in text:
        return list(_NORDIC_LOCATIONS)
    return [
        loc for loc in ALLOWED_LOCATIONS
        if re.search(rf"(?<!\w){re.escape(loc.casefold())}(?!\w)", text)
    ]


def _query_numeric_constraints(raw_query: str) -> dict[str, int | list[str]]:
    """Extract unambiguous numeric constraints as a deterministic guardrail."""
    text = raw_query.casefold().replace(",", "")
    out: dict[str, int | list[str]] = {}
    match = re.search(r"(?:more than|over)\s+(\d+)\s+employees", text)
    if match:
        out["employee_min"] = int(match.group(1)) + 1
    match = re.search(r"(?:at least|minimum of)\s+(\d+)\s+employees", text)
    if match:
        out["employee_min"] = int(match.group(1))
    match = re.search(r"(?:fewer than|under)\s+(\d+)\s+employees", text)
    if match:
        out["employee_max"] = max(0, int(match.group(1)) - 1)
    match = re.search(r"(?:founded\s+after\s+|post[- ]?)(\d{4})", text)
    if match:
        out["founded_after"] = int(match.group(1)) + 1
    match = re.search(r"founded\s+before\s+(\d{4})", text)
    if match:
        out["founded_before"] = int(match.group(1)) - 1

    compact = text.replace("$", "").replace("€", "").replace(" ", "")
    for bucket in ALLOWED_REVENUE_BUCKETS:
        if bucket.casefold() in compact:
            out["revenue_buckets"] = [bucket]
            break
    return out


def _fallback_keywords(raw_query: str) -> list[str]:
    text = raw_query.casefold()
    found = [phrase for phrase in _KEYWORD_PHRASES if phrase in text]
    if "ai-driven" in text or re.search(r"\bai\b", text):
        found.append("AI")
    return list(dict.fromkeys(found))


def sanitize_mandate(
    mandate: ParsedMandate, raw_query: str | None = None
) -> ParsedMandate:
    """Return a normalized copy while preserving explicit user constraints.

    Closed-vocabulary values are sanitized first. When the raw query contains
    an unambiguous domain, location, or numeric constraint, that literal user
    input wins over an LLM inference. This prevents terms such as "machine
    learning" from silently changing an explicit automotive mandate into the
    generic Technology industry.
    """
    f = mandate.filters
    new_industries: list[str] = []
    seen: set[str] = set()
    for tok in f.industries or []:
        n = _normalize_ind(tok)
        if n and n not in seen:
            new_industries.append(n)
            seen.add(n)

    new_locations: list[str] = []
    seen = set()
    for tok in f.locations or []:
        ns = _normalize_loc(tok)
        if ns is None:
            continue
        for n in ns:
            if n not in seen:
                new_locations.append(n)
                seen.add(n)

    new_rev: list[str] = []
    seen = set()
    for tok in f.revenue_buckets or []:
        n = _normalize_rev(tok)
        if n and n not in seen:
            new_rev.append(n)
            seen.add(n)

    overrides = _query_numeric_constraints(raw_query or "")
    explicit_industries = _explicit_industries(raw_query or "")
    explicit_locations = _explicit_locations(raw_query or "")

    cleaned = f.model_copy(
        update={
            "industries": new_industries,
            "locations": new_locations,
            "revenue_buckets": new_rev,
        }
    )
    if explicit_industries:
        cleaned = cleaned.model_copy(update={"industries": explicit_industries})
    if explicit_locations:
        cleaned = cleaned.model_copy(update={"locations": explicit_locations})
    if overrides:
        cleaned = cleaned.model_copy(update=overrides)
    explicit_keywords = _fallback_keywords(raw_query or "")
    industry_terms = {industry.casefold() for industry in cleaned.industries}
    semantic_keywords = [
        keyword
        for keyword in [*cleaned.keywords, *explicit_keywords]
        if keyword.casefold() not in industry_terms
    ]
    cleaned = cleaned.model_copy(
        update={"keywords": list(dict.fromkeys(semantic_keywords))}
    )
    return mandate.model_copy(update={"filters": cleaned})


def fallback_mandate(raw_query: str) -> ParsedMandate:
    """Conservative deterministic fallback used only when the LLM is unavailable."""
    numeric = _query_numeric_constraints(raw_query)
    filters = FilterSpec(
        industries=_explicit_industries(raw_query),
        locations=_explicit_locations(raw_query),
        keywords=_fallback_keywords(raw_query),
        **numeric,
    )
    return ParsedMandate(
        intent=raw_query.strip(),
        filters=filters,
        must_haves=[],
        # A partial regex parse cannot prove it captured every requirement.
        # Keep the complete query for validation instead of silently accepting
        # structured matches when an unfamiliar semantic condition was lost.
        semantic_requirements=[raw_query.strip()],
    )
