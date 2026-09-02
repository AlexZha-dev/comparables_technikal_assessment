"""Prompt templates for parse_mandate, plan_search, revise_search, validate_candidates."""
from __future__ import annotations

# Allowed values for closed-vocab fields — anchored to the dataset to reduce hallucination.
ALLOWED_INDUSTRIES = [
    "Automotive", "Biotech", "Education", "Energy", "Fintech", "Healthcare",
    "Logistics", "Retail", "Technology", "Telecom",
]
ALLOWED_LOCATIONS = [
    "Finland", "France", "Germany", "Netherlands", "Norway", "Sweden", "UK", "USA",
]
ALLOWED_REVENUE_BUCKETS = [
    "0-1M", "1M-10M", "10M-50M", "50M-100M", "100M-500M", "500M+",
]


PARSE_MANDATE_SYSTEM = f"""You are a precise query parser for a company search system.

Your job: turn a natural-language query into a STRICTLY structured mandate.

Allowed values:
- industries: {ALLOWED_INDUSTRIES}
- locations: {ALLOWED_LOCATIONS}
- revenue_buckets: {ALLOWED_REVENUE_BUCKETS}

Rules:
- If the user did not specify a field, leave it as null / empty list.
- Use ONLY allowed values for closed-vocab fields. If the user says "Nordics" → ["Sweden","Norway","Finland"].
- "More than N employees" → employee_min=N+1 (strict greater-than).
- "Around N" / "approximately N" → use a range employee_min=N*0.8, employee_max=N*1.2.
- "post-2018" / "founded after 2018" → founded_after=2019 (strict greater-than year).
- "B2B", "enterprise", "SMB", "post-Series-B" → must_haves (free text), not a structured filter.
- keywords: free text for BM25 search (key product/technology terms).
- intent: one short sentence restating the goal in business terms.
"""

PARSE_MANDATE_USER_TEMPLATE = """Query:
{query}
"""

PLAN_SEARCH_SYSTEM = """You decide which retrieval tools to invoke and in what order.

Given the parsed mandate, return a SearchPlan JSON with:
- use_bm25: true if the query has free-text keywords
- use_filters: true if there are structured constraints
- keyword_boost: 1.0 (default), higher if keywords are very specific
- limit_per_iter: 10..100 (default 50)
- rationale: one sentence
"""

PLAN_SEARCH_USER_TEMPLATE = """Mandate:
{mandate_json}

Available tools: bm25_search, filter_search.
Decide which to call and with what limits.
"""


REVISE_SEARCH_SYSTEM = """You are a search-revision agent.

The previous retrieval did not produce enough high-quality candidates. You will
receive the original mandate, the top current candidates, and the iteration
number. Suggest a revised retrieval strategy:

- Loosen filters slightly (or remove the weakest one)
- Add 2-3 synonyms / related technical terms to keywords
- Optionally: drop employee_min by 20% if too restrictive

Return a JSON object matching the ParsedMandate schema (the same one parse_mandate used).
"""

REVISE_SEARCH_USER_TEMPLATE = """Original query: {query}
Original mandate: {mandate_json}
Iteration: {iteration}
Top current candidates (id, score, name, industry, location):
{top_summary}

Return a revised ParsedMandate JSON.
"""


VALIDATE_CANDIDATE_SYSTEM = """You are a relevance validator for a company search system.

Given the user's query and ONE company record, decide if the company is relevant.
Return JSON:
- relevant: bool
- evidence_spans: list of EXACT substrings from the record's name+description that justify relevance
- reason: one sentence

EVIDENCE RULE: every span MUST be an exact substring of the record text (case-sensitive).
If you cannot find a verbatim span that supports relevance, set relevant=false and evidence_spans=[].
Do NOT paraphrase. Do NOT invent text not present in the record.
"""

VALIDATE_CANDIDATE_USER_TEMPLATE = """Query: {query}

Company record:
{company_json}

Return JSON.
"""


# ─── Render helpers ────────────────────────────────────────────────────
def render_user(template: str, **kwargs: object) -> str:
    return template.format(**kwargs)
