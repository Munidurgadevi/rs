# AI Candidate Discovery & Ranking System - Technical Presentation

## Architecture Overview

### Pipeline (`fast_rank.py`)
1. **Candidate Loading** - Reads JSONL/gzipped JSONL input
2. **Disqualifier Gates** - Hard exclusions for non-technical, consulting-only, CV/Speech/Robotics primary, LangChain-side-project-primary, honeypot language
3. **Feature Extraction** - Skill matching, ranking/search production evidence, production ML signals, evaluation framework detection, company quality scoring
4. **Behavioral Signal Scoring** - 23 Redrob signals (response rate, open-to-work, notice period, profile quality, engagement, location)
5. **Experience Bell Curve** - 5–9 years ideal multiplier
6. **Weighted Scoring** - 7 JD-aligned components combined multiplicatively
7. **Behavioral Penalties** - Low response rate, inactivity, high notice period
8. **Top-100 Selection** - Monotonic non-increasing scores with candidate_id tie-breaking

### Scoring Formula

| Component | Weight | Description |
|-----------|--------|-------------|
| Ranking/Search Production | 25% | Production ranking, search infra, retrieval systems |
| Company Quality | 20% | Product company vs consulting experience |
| Production ML | 15% | Deployed models, real users, latency |
| Evaluation Framework | 15% | NDCG/MRR/MAP/A/B test evidence |
| Skill Match | 10% | AI/ML core skills + Python |
| Behavioral Signals | 10% | Recruiter engagement, profile activity |
| Location Preference | 5% | India-based bonus |

### Key Optimizations

- **Single-pass processing** - Native Python loops over JSONL, ~32s for 100K candidates
- **CPU only** - No GPU, no external APIs, no model downloads
- **Deterministic tie-breaking** - Candidate ID ascending, monotonic score enforcement
- **Honeypot detection** - Pattern-based exclusion of "AI enthusiast" trap language

## Constraints Met

- ✅ CPU only (no GPU)
- ✅ No external API calls
- ✅ ~32 seconds runtime on 100K candidates (8-core CPU, 16GB RAM)
- ✅ Python 3.11+
- ✅ No pre-computation required

Top candidate: CAND_0011162 - "Recommendation Systems Engineer 5yr, 6 AI skills, response 75%, open; strong fit"
- 5 years experience (within target range)
- 6 core AI skills (embedding, retrieval, etc.)
- High recruiter response rate
- Open to work signal

Final score range: 0.78 - 0.86 (strong differentiation)