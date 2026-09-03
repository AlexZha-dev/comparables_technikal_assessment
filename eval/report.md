# Evaluation Report

_Generated: 2026-09-01 20:53:20_

Total queries: **5** · Passed: **5** / 5


## Summary

| ID | Pass | Latency (ms) | LLM calls | Iters | Results | Top-1 |
|----|------|-------------:|----------:|------:|--------:|-------|
| q1 | ✅ | 58079 | 3 | 1 | 10 | Nordic Fintech Solutions |
| q2 | ✅ | 49223 | 3 | 1 | 10 | Blair PLC |
| q3 | ✅ | 46286 | 3 | 1 | 10 | Brooks, Lam and Hayes |
| q4 | ✅ | 47919 | 3 | 1 | 1 | Hill-Ward |
| q5 | ✅ | 33363 | 4 | 2 | 1 | Helix BioCompute |

## Per-query details

### q1: Find AI-driven fintech companies in the Nordics with more than 100 employees
- **Passed**: ✅
- **Run id**: `1873d1df1906`
- **Mandate filters**: `{"industries": ["Fintech"], "locations": ["Finland", "Norway", "Sweden"], "employee_min": 101, "employee_max": null, "revenue_buckets": [], "founded_after": null, "founded_before": null, "keywords": ["AI-driven", "fintech"]}`
  - ✅ **parse_ok**: ok
  - ✅ **filters_ok**: ok
  - ✅ **in_dataset**: ok
  - ✅ **count_ok**: ok
  - ✅ **evidence_grounded**: ok
  - ✅ **llm_budget_ok**: ok
  - ✅ **iterations_ok**: ok
  - ✅ **revised_ok**: ok
  - ✅ **min_results_ok**: ok
- **Top results**:
  - [1] Nordic Fintech Solutions (Fintech, Finland, emp=240) score=1.000 relevant=True ev=6
  - [346] Davis PLC (Fintech, Finland, emp=4297) score=0.454 relevant=True ev=3
  - [2] Baltic Payments Cloud (Fintech, Finland, emp=420) score=0.300 relevant=True ev=2
  - [38] Wilkerson-Day (Fintech, Finland, emp=4824) score=0.300 relevant=True ev=2
  - [80] Novak and Sons (Fintech, Norway, emp=878) score=0.300 relevant=True ev=2

### q2: Renewable energy startups in Germany founded after 2018
- **Passed**: ✅
- **Run id**: `1873d1df1906`
- **Mandate filters**: `{"industries": ["Energy"], "locations": ["Germany"], "employee_min": null, "employee_max": null, "revenue_buckets": [], "founded_after": 2019, "founded_before": null, "keywords": ["renewable", "energy", "startups"]}`
  - ✅ **parse_ok**: ok
  - ✅ **filters_ok**: ok
  - ✅ **in_dataset**: ok
  - ✅ **count_ok**: ok
  - ✅ **evidence_grounded**: ok
  - ✅ **llm_budget_ok**: ok
  - ✅ **iterations_ok**: ok
  - ✅ **revised_ok**: ok
  - ✅ **min_results_ok**: ok
- **Top results**:
  - [23] Blair PLC (Energy, UK, emp=1343) score=0.700 relevant=False ev=0
  - [81] Russell Group (Energy, Netherlands, emp=999) score=0.700 relevant=False ev=0
  - [165] Manning Group (Energy, Sweden, emp=522) score=0.700 relevant=False ev=0
  - [225] Stewart Ltd (Energy, UK, emp=3643) score=0.700 relevant=False ev=0
  - [265] Bailey-Cook (Energy, UK, emp=2839) score=0.700 relevant=False ev=0

### q3: Healthcare companies with $50M-$100M revenue in the USA
- **Passed**: ✅
- **Run id**: `1873d1df1906`
- **Mandate filters**: `{"industries": ["Healthcare"], "locations": ["USA"], "employee_min": null, "employee_max": null, "revenue_buckets": ["50M-100M"], "founded_after": null, "founded_before": null, "keywords": []}`
  - ✅ **parse_ok**: ok
  - ✅ **filters_ok**: ok
  - ✅ **in_dataset**: ok
  - ✅ **count_ok**: ok
  - ✅ **evidence_grounded**: ok
  - ✅ **llm_budget_ok**: ok
  - ✅ **iterations_ok**: ok
  - ✅ **revised_ok**: ok
  - ✅ **min_results_ok**: ok
- **Top results**:
  - [63] Brooks, Lam and Hayes (Healthcare, USA, emp=2902) score=0.300 relevant=True ev=1
  - [217] Miller Ltd (Healthcare, USA, emp=4575) score=0.300 relevant=True ev=1
  - [550] Jones, Compton and Day (Healthcare, USA, emp=4267) score=0.300 relevant=True ev=1
  - [922] Fisher, Payne and Thompson (Healthcare, USA, emp=135) score=0.300 relevant=True ev=1
  - [1424] Gonzalez, Smith and Padilla (Healthcare, USA, emp=4769) score=0.300 relevant=True ev=1

### q4: Autonomous driving and machine learning companies in the Netherlands
- **Passed**: ✅
- **Run id**: `1873d1df1906`
- **Mandate filters**: `{"industries": ["Technology"], "locations": ["Netherlands"], "employee_min": null, "employee_max": null, "revenue_buckets": [], "founded_after": null, "founded_before": null, "keywords": ["autonomous driving", "machine learning"]}`
  - ✅ **parse_ok**: ok
  - ✅ **filters_ok**: ok
  - ✅ **in_dataset**: ok
  - ✅ **count_ok**: ok
  - ✅ **evidence_grounded**: ok
  - ✅ **llm_budget_ok**: ok
  - ✅ **iterations_ok**: ok
  - ✅ **revised_ok**: ok
  - ✅ **min_results_ok**: ok
- **Top results**:
  - [1337] Hill-Ward (Automotive, Netherlands, emp=3337) score=0.700 relevant=True ev=1

### q5: Biotech in France with more than 5000 employees
- **Passed**: ✅
- **Run id**: `1873d1df1906`
- **Mandate filters**: `{"industries": ["Biotech", "Healthcare"], "locations": ["France"], "employee_min": 5001, "employee_max": null, "revenue_buckets": [], "founded_after": null, "founded_before": null, "keywords": ["biotech", "biopharma", "pharma", "life sciences", "biomedical"]}`
  - ✅ **parse_ok**: ok
  - ✅ **filters_ok**: ok
  - ✅ **in_dataset**: ok
  - ✅ **count_ok**: ok
  - ✅ **evidence_grounded**: ok
  - ✅ **llm_budget_ok**: ok
  - ✅ **iterations_ok**: ok
  - ✅ **revised_ok**: ok
- **Top results**:
  - [4] Helix BioCompute (Biotech, Germany, emp=95) score=0.700 relevant=False ev=0
