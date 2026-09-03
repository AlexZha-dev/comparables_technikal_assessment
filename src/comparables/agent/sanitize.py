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

from comparables.agent.prompts import (
    ALLOWED_INDUSTRIES,
    ALLOWED_LOCATIONS,
    ALLOWED_REVENUE_BUCKETS,
)
from comparables.schemas.mandate import ParsedMandate

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

_LOCATION_SYNONYMS: list[tuple[str, str]] = [
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


def sanitize_mandate(mandate: ParsedMandate) -> ParsedMandate:
    """Return a new ParsedMandate with hallucinated industry/location/revenue values stripped."""
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

    # Build a copy of the mandate with the cleaned filters.
    from copy import replace

    return replace(mandate, filters=replace(f,
        industries=new_industries,
        locations=new_locations,
        revenue_buckets=new_rev,
    ))
