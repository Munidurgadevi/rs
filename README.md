# AI Candidate Discovery & Ranking System

Production-grade candidate ranking system for the Redrob Hackathon Senior AI Engineer role.

## Architecture

```
fast_rank.py                  # Main entry point - optimized standalone script (~32s)
                           # Includes all disqualifier gates, evaluation framework,
                           # experience bell curve, and behavioral signal scoring

improve_ranking.py           # Full evidence-extraction ranking with detailed analysis
                           # Uses career-history text parsing for production evidence

ai_ranking_system/           # Reference package with modular components
├── disqualifier_gates.py    # Hard exclusions for CV, consulting, LangChain-only, etc.
├── evaluation_framework.py  # NDCG, MRR, MAP computation and scoring
├── vector_db_simulator.py   # FAISS + TF-IDF hybrid search simulation
├── learning_to_rank.py      # XGBoost/GradientBoosting LTR model
├── feature_extractor.py     # Production ML, search, ranking, company quality features
├── behavioral_scorer.py     # Redrob signal scoring (response rate, activity, location)
├── honeypot_detector.py     # Suspicious profile detection
├── embedding_ranker.py      # Sentence-transformers semantic similarity
├── semantic_matcher.py      # TF-IDF fallback for semantic matching
├── reasoning_generator.py   # Evidence-based human-readable explanations
├── main_pipeline.py         # Full pipeline orchestration
├── rank.py                  # Alternative entry point (python -m ai_ranking_system.rank)
└── requirements.txt         # Dependencies
```

**Note**: `fast_rank.py` is a self-contained standalone script that implements all core logic inline for maximum performance. The `ai_ranking_system/` package contains modular reference implementations of advanced components (LTR, vector DB, evaluation frameworks) that can be integrated for enhanced ranking quality.

## Scoring Formula

```
Final Score = 0.25 * Ranking/Search Production (ranking systems + search infra + production context) +
               0.20 * Company Quality (product vs consulting experience) +
               0.15 * Production ML Experience (deployed, real users, latency) +
               0.15 * Evaluation Framework (NDCG/MRR/MAP/A/B test evidence) +
               0.10 * Skill Match (AI/ML core skills + Python) +
               0.10 * Behavioral Signals (response rate, profile quality, activity) +
               0.05 * Location Preference (India-based bonus)

With disqualifier gates applied before scoring:
- Non-technical titles without production IR/Search/Ranking
- Civil/Mechanical Engineers without software/ML production
- CV/Speech/Robotics primary without NLP/IR production
- Consulting-only backgrounds
- LangChain/OpenAI side-project-primary profiles
- "AI enthusiast" / "curious about AI" / "getting into AI" trap language
- Pure research without production deployment
- Data/Backend Engineers transitioning to ML without production AI
- Extreme job hoppers (<12 months average tenure with 4+ roles)

Experience multiplier (bell curve):
- 5-9 years: 1.0x (ideal)
- 4-5 or 9-12 years: 0.85x
- 3-4 or 12-15 years: 0.6x
- Outside range: 0.3x
```

## Features

- **Ranking/Search Production**: Detects production ranking/search experience with infrastructure evidence (pinecone, milvus, faiss, elasticsearch)
- **Company Quality**: Product company experience weighting with consulting firm penalties
- **Evaluation Framework**: NDCG/MRR/MAP/A/B test evidence detection
- **Skill Match**: AI/ML core skills detection (embedding, retrieval, ranking, vector search, NLP, RAG)
- **Behavioral Signals**: All 23 Redrob signals via BehavioralSignalScorer
- **Location Preference**: India-based candidates receive preference
- **Honeypot Detection**: Traps "AI enthusiast", "curious about AI" language

## How to Run

```bash
# Requires Python 3.11+
python --version  # Should show Python 3.11.x

# Install dependencies (minimal - fast_rank.py uses only Python stdlib + local behavioral_scorer)
pip install -r requirements.txt

# Fast version (no model download, ~32 seconds on CPU)
# Produces submission.csv from candidates.jsonl
py fast_rank.py --candidates candidates.jsonl --out team_id.csv
```

## Constraints

- **Python**: 3.11+
- **Compute**: CPU only (no GPU)
- **Network**: No external API calls during ranking
- **Runtime**: ~32 seconds for 100K candidates on 8-core CPU, 16GB RAM
- **Pre-computation**: None required
- **Dependencies**: Pure Python / standard library; local `ai_ranking_system` package

## Output Format

```
candidate_id,rank,score,reasoning
CAND_0046064,1,0.85967,"Top-tier candidate. Senior NLP Engineer — 8.9 years experience; 4 core AI skills (Pinecone, OpenSearch, Elasticsearch); production Pinecone experience; high recruiter response (78%)."
CAND_0008425,2,0.8525,"Top-tier candidate. Senior NLP Engineer — 7.8 years experience; 5 core AI skills (pgvector, NLP, Semantic Search); production Pinecone experience; response rate 66%."
CAND_0018499,3,0.85249,"Top-tier candidate. Senior Machine Learning Engineer — 7.2 years experience; 6 core AI skills (Recommendation Systems, Pinecone, Information Retrieval); production Milvus experience; response rate 61%."
CAND_0002025,4,0.844,"Top-tier candidate. Senior AI Engineer — 5.9 years experience; 6 core AI skills (FAISS, OpenSearch, NLP); production Faiss experience; high recruiter response (80%)."

