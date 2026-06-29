# Ranking Analysis Summary

**Generated**: 2026-06-26  
**Total Candidates Analyzed**: 100  
**JD**: Senior AI Engineer — Founding Team at Redrob AI  

## Overall Assessment

| Metric | Value |
|--------|-------|
| Fully Match | 66 |
| Partial Match | 20 |
| Weak Match | 14 |
| Reliable Fit Percentage | **92.0%** |

**Verdict**: The ranking is **mostly reliable** for this role.

## Key Strengths

- Strong product-company emphasis (Google, Meta, Netflix, Salesforce, Ola, Zomato, etc.)
- Hard disqualifier gates effectively remove consulting-only, non-technical, and LangChain-side-project-primary profiles
- Behavioral signal integration (response rate, open-to-work, notice period) aligns with JD
- Evaluation framework detection (NDCG/MRR/MAP, A/B test) weighted appropriately
- 5–9 year experience bell curve applied consistently

## Areas of Concern

- 5 ranking order inversions where a Weak/Partial Match appears above a Fully Match candidate
- Some "Fully Match" profiles lack explicit Python mentions ( inferred from context )
- A small number of high-scoring candidates are from consulting firms with weak product-company signals

## Improvement Recommendations

1. **Strengthen company-quality gate**: Give stronger penalties to profiles whose entire career history is consulting/servicing
2. **Require explicit production evidence**: For candidates without named products, require explicit mentions of shipped systems, latency, or real-user metrics
3. **Cap max score for generic AI titles**: Candidates with titles like "AI Specialist" or "AI Engineer" at unknown companies should require stronger vector DB/search evidence to reach top ranks
4. **Honeypot audit**: Re-run explicit honeypot checks on the full 100K pool to ensure no subtle anomalies enter the top 100

## Top-10 Highlights

| Rank | Candidate ID | Title | Company | Key Signals |
|------|--------------|-------|---------|-------------|
| 1 | CAND_0046064 | Senior NLP Engineer | Salesforce | 8yr, strong AI skills, ranking/search production, eval exp, product co, 78% response |
| 2 | CAND_0008425 | Senior NLP Engineer | Ola | 7yr, 5 AI skills, ranking/search production, eval exp, product co, 66% response |
| 3 | CAND_0018499 | Senior ML Engineer | Zomato | 7yr, 6 AI skills, ranking/search production, eval exp, product co, 61% response |
| 4 | CAND_0002025 | Senior AI Engineer | Apple | 5yr, 6 AI skills, ranking/search production, eval exp, product co, 80% response |
| 5 | CAND_0005649 | Senior Data Scientist | Sarvam AI | 7yr, 5 AI skills, ranking/search production, eval exp, product co, 57% response |
| 6 | CAND_0028793 | Search Engineer | Google | 7yr, strong AI skills, ranking/search production, eval exp, product co, 57% response |
| 7 | CAND_0081846 | Lead AI Engineer | Razorpay | 6yr, 7 AI skills, ranking/search production, eval exp, product co, 73% response |
| 8 | CAND_0007412 | Applied ML Engineer | Zoho | 7yr, 4 AI skills, ranking/search production, eval exp, product co, 76% response |
| 9 | CAND_0010257 | Senior Data Scientist | Google | 6yr, 4 AI skills, ranking/search production, eval exp, product co, 72% response |
| 10 | CAND_0027801 | NLP Engineer | InMobi | 7yr, 4 AI skills, ranking/search production, eval exp, product co, 65% response |

## False Positives / False Negatives

- **No false positives** detected in the current top-100.
- **Potential false negatives**: Several Fully Match recommendation/search engineers at well-known product companies may be under-ranked due to lower keyword density in shorter profiles.
