#!/usr/bin/env python3
"""
JD-Compliant Candidate Ranking System for Senior AI Engineer Role.
Extracts specific evidence from candidate profiles and applies JD rules.
"""

import json
import csv
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ============================================================
# Configuration
# ============================================================
CANDIDATES_FILE = Path("candidates.jsonl")
SUBMISSION_FILE = Path("submission.csv")
OUTPUT_FILE = Path("submission_improved.csv")

# ============================================================
# JD Rules and Patterns
# ============================================================
CONSULTING_FIRMS = {
    'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini',
    'mindtree', 'tech mahindra', 'hcl', 'lti', 'persistent'
}

NON_TECH_TITLES = {
    'marketing manager', 'sales', 'operations manager', 'project manager',
    'hr manager', 'accountant', 'content writer', 'graphic designer',
    'civil engineer', 'mechanical engineer', 'customer support',
    'business analyst', 'sales executive', 'operations'
}

CV_SPEECH_ROBOTICS = {
    'computer vision', 'cv engineer', 'vision engineer', 'robotics',
    'speech engineer', 'asr engineer', 'tts engineer'
}

# Evidence extraction patterns - MUST find explicit evidence
EVIDENCE_PATTERNS = {
    'embeddings_retrieval': [
        r'sentence-transformers?', r'\bBGE\b', r'\bE5\b', r'openai embeddings',
        r'embedding model', r'embedding generation', r'fine-tuned embedding',
        r'embedding fine-tuning', r'embedding drift', r'index refresh'
    ],
    'vector_db': [
        r'\bPinecone\b', r'\bWeaviate\b', r'\bQdrant\b', r'\bMilvus\b', r'\bFAISS\b',
        r'\bElasticsearch\b', r'\bOpenSearch\b', r'\bpgvector\b', r'vector database',
        r'vector index', r'vector store'
    ],
    'hybrid_search': [
        r'hybrid search', r'hybrid retrieval', r'sparse.*dense', r'dense.*sparse',
        r'bm25.*embedding', r'keyword.*semantic', r'tf-idf.*embedding'
    ],
    'ranking_systems': [
        r'ranking system', r'ranking model', r'ranking layer', r'ranking pipeline',
        r'search engine', r'search system', r'search pipeline',
        r'recommendation system', r'recommendation engine', r'recommendation model',
        r'recommendation feed', r'discovery feed', r'retrieval system',
        r'reranking', r're-ranking', r'learning.to.rank', r'\bLTR\b',
        r'XGBoost.*rank', r'Lambda?MART', r'RankNet', r'coordinate ascent'
    ],
    'eval_framework': [
        r'\bNDCG\b', r'\bMRR\b', r'\bMAP\b', r'precision@k', r'recall@k',
        r'a/b test', r'ab test', r'evaluation framework', r'eval harness',
        r'eval pipeline', r'offline.to.online', r'offline.*online correlation',
        r'relevance judgment', r'human label', r'online experiment',
        r'statistical significance', r'confidence interval'
    ],
    'production_ml': [
        r'production', r'deployed', r'shipped', r'live', r'real.users',
        r'real.world', r'latency', r'throughput', r'serving', r'kubernetes',
        r'\bK8s\b', r'BentoML', r'MLflow', r'Kubeflow', r'monitoring',
        r'alerting', r'observability', r'p95', r'p99', r'scale\b',
        r'microservice', r'api\s+serving', r'model serving'
    ],
    'llm_experience': [
        r'\bLLM\b', r'\bGPT\b', r'\bLLaMA\b', r'\bMistral\b', r'\bGemma\b',
        r'\bQwen\b', r'fine-tuning', r'\bLoRA\b', r'\bQLoRA\b', r'\bPEFT\b',
        r'\bRAG\b', r'retrieval augmented', r'prompt engineering',
        r'generative ai', r'\bGenAI\b'
    ],
    'python': [
        r'\bPython\b', r'\bPyTorch\b', r'\bTensorFlow\b', r'scikit-learn',
        r'\bnumpy\b', r'\bpandas\b', r'FastAPI', r'Flask', r'Django'
    ]
}

# Strong product companies (weighted higher)
STRONG_PRODUCT = {
    'google', 'amazon', 'microsoft', 'apple', 'meta', 'netflix', 'uber', 'adobe',
    'salesforce', 'ola', 'zomato', 'swiggy', 'flipkart', 'paytm', 'phonepe',
    'dream11', 'meesho', 'inmobi', 'freshworks', 'zoho', 'razorpay', 'linkedin',
    'airbnb', 'spotify', 'stripe', 'shopify', 'bytedance', 'tencent', 'alibaba'
}

GOOD_PRODUCT = {
    'cred', 'groww', 'policybazaar', 'nykaa', 'hasura', 'postman', 'browserstack',
    'hotstar', 'pharmeasy', 'aganitha', 'verloop.io', 'repharse.ai', 'sarvam ai',
    'yellow.ai', 'pied piper', 'hooli', 'wayne enterprises', 'acme corp',
    'globex inc', 'initech', 'dunder mifflin', 'stark industries'
}

PREFERRED_LOCATIONS = {
    'pune', 'noida', 'hyderabad', 'mumbai', 'delhi', 'ncr', 'bangalore', 'gurgaon',
    'chennai', 'kolkata', 'vizag', 'trivandrum'
}


# ============================================================
# Evidence Extraction Functions
# ============================================================
def extract_evidence(text: str, patterns: list) -> List[str]:
    """Extract specific evidence from text using regex patterns."""
    evidence = []
    for p in patterns:
        matches = re.findall(p, text, re.IGNORECASE)
        if matches:
            evidence.extend(matches[:2])  # Max 2 per pattern
    return list(set(evidence))


def extract_career_details(candidate: Dict) -> Dict:
    """Extract structured details from career history."""
    career = candidate.get('career_history', [])
    details = {
        'companies': [],
        'titles': [],
        'descriptions': [],
        'has_search_ranking_production': False,
        'has_embeddings_production': False,
        'has_vector_db': False,
        'has_eval_framework': False,
        'has_llm': False,
        'production_evidence': [],
        'search_ranking_evidence': [],
        'eval_evidence': [],
        'embedding_evidence': [],
        'llm_evidence': [],
    }
    
    for role in career:
        company = role.get('company', '').lower()
        title = role.get('title', '').lower()
        desc = role.get('description', '').lower()
        
        details['companies'].append(company)
        details['titles'].append(title)
        details['descriptions'].append(desc)
        
        # Check for production search/ranking in description
        if re.search(r'ranking|search.*system|recommendation.*system|retrieval.*system|discovery.*feed', desc):
            details['has_search_ranking_production'] = True
            details['search_ranking_evidence'].append(f"{role.get('company', '')}: {role.get('title', '')}")
        
        # Check for embeddings
        if re.search(r'sentence.transformers?|bge|e5|embedding', desc):
            details['has_embeddings_production'] = True
            details['embedding_evidence'].append(f"{role.get('company', '')}: {role.get('title', '')}")
        
        # Check for vector DB
        if re.search(r'pinecone|weaviate|qdrant|milvus|faiss|elasticsearch|opensearch|pgvector', desc):
            details['has_vector_db'] = True
        
        # Check for evaluation framework
        if re.search(r'ndcg|mrr|map|a/b test|evaluation framework|eval harness|offline.*online', desc):
            details['has_eval_framework'] = True
            details['eval_evidence'].append(f"{role.get('company', '')}: {role.get('title', '')}")
        
        # Check for LLM
        if re.search(r'llm|gpt|llama|mistral|fine-tuning|lora|qlora|peft|rag', desc):
            details['has_llm'] = True
            details['llm_evidence'].append(f"{role.get('company', '')}: {role.get('title', '')}")
        
        # Production evidence
        if re.search(r'production|deployed|shipped|live|real.users|latency|throughput|serving|kubernetes', desc):
            details['production_evidence'].append(f"{role.get('company', '')}: {role.get('title', '')}")
    
    return details


def extract_skill_evidence(skills: List[Dict]) -> Dict:
    """Extract evidence from skills list."""
    skill_names = [s.get('name', '').lower() for s in skills]
    skill_details = {
        'has_python': False,
        'has_embeddings': False,
        'has_vector_db': False,
        'has_ranking': False,
        'has_eval': False,
        'has_llm': False,
        'python_skills': [],
        'ai_skills': []
    }
    
    for skill in skills:
        name = skill.get('name', '').lower()
        proficiency = skill.get('proficiency', '')
        
        if 'python' in name or 'pytorch' in name or 'tensorflow' in name or 'scikit-learn' in name:
            skill_details['has_python'] = True
            skill_details['python_skills'].append(skill.get('name', ''))
        
        if any(t in name for t in ['embedding', 'sentence-transformers', 'bge', 'e5']):
            skill_details['has_embeddings'] = True
            skill_details['ai_skills'].append(skill.get('name', ''))
        
        if any(t in name for t in ['vector', 'pinecone', 'weaviate', 'qdrant', 'milvus', 'faiss', 'elasticsearch', 'opensearch']):
            skill_details['has_vector_db'] = True
            skill_details['ai_skills'].append(skill.get('name', ''))
        
        if any(t in name for t in ['ranking', 'recommendation', 'search', 'learning-to-rank', 'retrieval']):
            skill_details['has_ranking'] = True
            skill_details['ai_skills'].append(skill.get('name', ''))
        
        if any(t in name for t in ['ndcg', 'mrr', 'map', 'evaluation', 'metric']):
            skill_details['has_eval'] = True
            skill_details['ai_skills'].append(skill.get('name', ''))
        
        if any(t in name for t in ['llm', 'gpt', 'llama', 'mistral', 'fine-tuning', 'lora', 'qlora', 'peft', 'rag']):
            skill_details['has_llm'] = True
            skill_details['ai_skills'].append(skill.get('name', ''))
    
    return skill_details


# ============================================================
# Disqualifier Gates
# ============================================================
def check_disqualifiers(candidate: Dict) -> Tuple[bool, str]:
    """
    Returns (is_disqualified, reason).
    These are HARD gates that eliminate candidates regardless of other strengths.
    """
    profile = candidate.get('profile', {})
    career = candidate.get('career_history', [])
    skills = candidate.get('skills', [])
    summary = profile.get('summary', '').lower()
    title = profile.get('current_title', '').lower()
    
    skill_names = [s.get('name', '').lower() for s in skills]
    career_details = extract_career_details(candidate)
    
    # Gate 1: Non-technical title with no production IR/Search/Ranking
    if any(t in title for t in NON_TECH_TITLES):
        if not career_details['has_search_ranking_production']:
            return True, f"Non-technical title ({profile.get('current_title', '')}) with no production search/ranking/retrieval experience"
    
    # Gate 2: Civil/Mechanical engineer primary without software/ML production
    non_software = ['civil engineer', 'mechanical engineer', 'electrical engineer', 'chemical engineer']
    if any(t in title for t in non_software):
        has_software = any(kw in summary for kw in ['python', 'ml', 'machine learning', 'deployed', 'production', 'software'])
        if not has_software:
            return True, f"Primary expertise is non-software engineering ({profile.get('current_title', '')}) without NLP/IR/Search/Ranking production"
    
    # Gate 3: CV/Speech/Robotics primary without NLP/IR production
    recent_cv = False
    if career:
        recent_title = career[0].get('title', '').lower()
        recent_cv = any(t in recent_title for t in CV_SPEECH_ROBOTICS)
    
    if recent_cv:
        # Must have explicit production search/ranking/retrieval in MOST RECENT role
        recent_desc = career[0].get('description', '').lower()
        recent_search_patterns = [
            r'search\s+(?:system|engine|pipeline|infrastructure)',
            r'ranking\s+(?:system|model|pipeline|layer)',
            r'retrieval\s+(?:system|pipeline)',
            r'recommendation\s+(?:system|engine|model|feed)',
            r'vector\s+(?:search|database|index|retrieval)',
            r'semantic\s+search', r'hybrid\s+search',
            r'learning.?to.?rank', r'\b(?:ndcg|mrr|map)\b',
            r'\b(?:pinecone|milvus|faiss|weaviate|qdrant|elasticsearch|opensearch)\b',
            r're-?ranking\s+(?:model|layer|system)',
            r're-?rank\s+(?:over|with|using)',
            r'collaborative\s+filtering.*(?:ranking|recommendation)',
        ]
        has_recent_search = any(re.search(p, recent_desc, re.IGNORECASE) for p in recent_search_patterns)
        if not has_recent_search:
            return True, f"Primary expertise is CV/Speech/Robotics ({profile.get('current_title', '')}) without production NLP/IR/Search/Ranking"
    
    # Gate 4: Consulting-only background
    consulting_only = True
    for role in career:
        company = role.get('company', '').lower()
        duration = role.get('duration_months', 0)
        if duration < 12:
            continue
        if company not in CONSULTING_FIRMS:
            consulting_only = False
            break
    if consulting_only:
        return True, "Consulting-only background with no product company experience"
    
    # Gate 5: LangChain/OpenAI side projects as primary experience
    side_project_patterns = [
        re.compile(r'\b(?:langchain|openai)\s+(?:api|tutorial|side project|experiment)\b', re.IGNORECASE),
        re.compile(r'\b(?:side project|online course|taking courses)\b.*\b(?:ai|ml|rag|llm)\b', re.IGNORECASE),
        re.compile(r'\b(?:curious about|exploring|getting into)\b.*\b(?:ai|ml|genai)\b', re.IGNORECASE),
    ]
    side_signals = sum(1 for p in side_project_patterns if p.search(summary))
    prod_signals = sum(1 for kw in ['deployed', 'production', 'shipped', 'live', 
                                     'real users', 'latency', 'throughput', 'a/b test']
                       if kw in summary)
    has_real_jobs = len([r for r in career if r.get('duration_months', 0) >= 12]) >= 1
    
    if side_signals >= 2 and prod_signals <= 1 and has_real_jobs:
        return True, "AI experience primarily LangChain/OpenAI side projects without substantial pre-LLM production ML"
    
    # Gate 6: Pure research without production
    research_titles = ['research scientist', 'research engineer', 'postdoc', 'research analyst']
    research_companies = ['university', 'institute', 'lab', 'academia']
    is_research = any(
        any(k in r.get('title', '').lower() for k in research_titles) or
        any(k in r.get('company', '').lower() for k in research_companies)
        for r in career[:2]
    )
    has_production = career_details['has_search_ranking_production'] or career_details['has_embeddings_production']
    if is_research and not has_production:
        return True, "Pure research background without production deployment"
    
    # Gate 7: Data/Backend Engineer transitioning to ML without production AI experience
    transition_titles = ['data engineer', 'backend engineer', 'software engineer', 'backend developer']
    current_title_lower = profile.get('current_title', '').lower()
    is_non_ai_title = any(t in current_title_lower for t in transition_titles)
    
    if is_non_ai_title:
        # Check if this is a genuine AI/ML role or just data/backend work
        ai_role_titles = ['ai engineer', 'ml engineer', 'machine learning engineer', 'data scientist',
                         'applied scientist', 'nlp engineer', 'computer vision engineer',
                         'deep learning engineer', 'research engineer', 'ai researcher']
        is_current_ai_role = any(t in current_title_lower for t in ai_role_titles)
        
        if not is_current_ai_role:
            # They're in a backend/data role. Check if they have PRODUCTION AI/ML experience
            has_ai_production = (career_details['has_search_ranking_production'] or 
                                career_details['has_embeddings_production'] or
                                career_details['has_llm'])
            
            # Also check if they're explicitly transitioning
            transitioning_keywords = [
                'transitioning toward', 'transition to ai', 'transition to ml',
                'moving into ai', 'moving into ml', 'shifting to ai', 'shifting to ml',
                'exploring ai', 'exploring ml', 'getting into ai', 'getting into ml',
                'self-directed ml projects', 'kaggle competitions', 'side projects.*fine-tuning',
                'want to do more of the ml', 'interested in transitioning'
            ]
            is_transitioning = any(re.search(k, summary, re.IGNORECASE) for k in transitioning_keywords)
            
            if not has_ai_production and is_transitioning:
                return True, f"Current role is {profile.get('current_title', '')} without production AI/ML experience; candidate is transitioning to ML"
    
    # Gate 8: Title-chaser check (mild penalty as disqualifier only if extreme)
    if len(career) >= 4:
        avg_tenure = sum(r.get('duration_months', 0) for r in career) / len(career)
        if avg_tenure < 12:
            return True, f"Extreme job hopping (avg {avg_tenure:.0f} months per role, {len(career)} roles)"
    
    return False, ""


# ============================================================
# Evidence-Based Scoring
# ============================================================
def compute_score(candidate: Dict) -> Tuple[float, str, Dict]:
    """
    Compute JD-compliant score with evidence extraction.
    Returns (score, reasoning, evidence_dict)
    """
    profile = candidate.get('profile', {})
    career = candidate.get('career_history', [])
    skills = candidate.get('skills', [])
    signals = candidate.get('redrob_signals', {})
    
    title = profile.get('current_title', '')
    years = profile.get('years_of_experience', 0)
    location = profile.get('location', '')
    
    full_text = ' '.join([
        profile.get('headline', ''),
        profile.get('summary', ''),
        ' '.join(r.get('description', '') + ' ' + r.get('title', '') for r in career),
        ' '.join(s.get('name', '') for s in skills)
    ]).lower()
    
    career_text = ' '.join(
        r.get('description', '') + ' ' + r.get('title', '') for r in career
    ).lower()
    
    skill_details = extract_skill_evidence(skills)
    career_details = extract_career_details(candidate)
    
    # Extract evidence
    evidence = {}
    for category, patterns in EVIDENCE_PATTERNS.items():
        evidence[category] = extract_evidence(full_text, patterns)
    
    # ===== SCORING =====
    score = 0.0
    score_breakdown = []
    
    # 1. Embeddings-based retrieval (MUST HAVE) - Weight: 15%
    if career_details['has_embeddings_production'] or skill_details['has_embeddings']:
        emb_score = 1.0
        score_breakdown.append(("Embeddings retrieval", 0.15, "Production embeddings system found"))
    elif evidence['embeddings_retrieval']:
        emb_score = 0.7
        score_breakdown.append(("Embeddings retrieval", 0.10, "Mentioned in profile"))
    else:
        emb_score = 0.0
    score += emb_score * 0.15
    
    # 2. Vector DB / Hybrid search (MUST HAVE) - Weight: 15%
    if career_details['has_vector_db'] or skill_details['has_vector_db']:
        vdb_score = 1.0
        score_breakdown.append(("Vector DB/Hybrid", 0.15, "Production vector DB experience"))
    elif evidence['vector_db'] or evidence['hybrid_search']:
        vdb_score = 0.7
        score_breakdown.append(("Vector DB/Hybrid", 0.10, "Mentioned in profile"))
    else:
        vdb_score = 0.0
    score += vdb_score * 0.15
    
    # 3. Ranking/Search/RecSys (MUST HAVE) - Weight: 20%
    if career_details['has_search_ranking_production']:
        rank_score = 1.0
        score_breakdown.append(("Ranking/Search/RecSys", 0.20, "Production ranking/search system"))
    elif skill_details['has_ranking']:
        rank_score = 0.8
        score_breakdown.append(("Ranking/Search/RecSys", 0.12, "Skills in ranking/search"))
    else:
        rank_score = 0.0
    score += rank_score * 0.20
    
    # 4. Evaluation frameworks (MUST HAVE) - Weight: 15%
    if career_details['has_eval_framework']:
        eval_score = 1.0
        score_breakdown.append(("Eval frameworks", 0.15, "Production eval framework (NDCG/MRR/MAP/A/B)"))
    elif skill_details['has_eval']:
        eval_score = 0.7
        score_breakdown.append(("Eval frameworks", 0.10, "Skills in evaluation"))
    elif evidence['eval_framework']:
        eval_score = 0.4
        score_breakdown.append(("Eval frameworks", 0.05, "Mentioned in profile"))
    else:
        eval_score = 0.0
    score += eval_score * 0.15
    
    # 5. Python (MUST HAVE) - Weight: 10%
    if skill_details['has_python']:
        py_score = 1.0
        score_breakdown.append(("Python", 0.10, "Python skills confirmed"))
    elif evidence['python']:
        py_score = 0.5
        score_breakdown.append(("Python", 0.05, "Mentioned"))
    else:
        py_score = 0.0
    score += py_score * 0.10
    
    # 6. Production ML deployment (MUST HAVE) - Weight: 10%
    if career_details['production_evidence']:
        prod_score = 1.0
        score_breakdown.append(("Production ML", 0.10, "Production deployment confirmed"))
    elif evidence['production_ml']:
        prod_score = 0.6
        score_breakdown.append(("Production ML", 0.05, "Mentioned"))
    else:
        prod_score = 0.0
    score += prod_score * 0.10
    
    # 7. Product company experience (MUST HAVE) - Weight: 10%
    product_months = 0
    total_months = 0
    for role in career:
        company = role.get('company', '').lower()
        duration = role.get('duration_months', 0)
        total_months += duration
        if company not in CONSULTING_FIRMS:
            product_months += duration
    
    if total_months > 0:
        product_ratio = product_months / total_months
        if product_ratio >= 0.7:
            prod_co_score = 1.0
            score_breakdown.append(("Product company", 0.10, f"{product_ratio:.0%} product company experience"))
        elif product_ratio >= 0.4:
            prod_co_score = 0.7
            score_breakdown.append(("Product company", 0.07, f"{product_ratio:.0%} product company"))
        else:
            prod_co_score = 0.3
            score_breakdown.append(("Product company", 0.03, f"{product_ratio:.0%} product company (mostly consulting)"))
    else:
        prod_co_score = 0.0
    score += prod_co_score * 0.10
    
    # 8. LLM experience (NICE TO HAVE) - Weight: 5%
    if career_details['has_llm'] or skill_details['has_llm']:
        llm_score = 1.0
        score_breakdown.append(("LLM exp", 0.05, "LLM fine-tuning/RAG experience"))
    elif evidence['llm_experience']:
        llm_score = 0.5
        score_breakdown.append(("LLM exp", 0.025, "LLM mentioned"))
    else:
        llm_score = 0.0
    score += llm_score * 0.05
    
    # ===== BONUSES AND PENALTIES =====
    
    # Experience range bonus (5-9 years preferred)
    if 5 <= years <= 9:
        score *= 1.0
        score_breakdown.append(("Experience range", 0.0, f"{years} years (ideal 5-9)"))
    elif 4 <= years < 5 or 9 < years <= 12:
        score *= 0.85
        score_breakdown.append(("Experience range", 0.0, f"{years} years (near target)"))
    elif 3 <= years < 4 or 12 < years <= 15:
        score *= 0.6
        score_breakdown.append(("Experience range", 0.0, f"{years} years (outside preferred range)"))
    else:
        score *= 0.3
    
    # Behavioral signals
    resp_rate = signals.get('recruiter_response_rate', 0.5)
    last_active = signals.get('last_active_date', '')
    open_to_work = signals.get('open_to_work_flag', False)
    notice = signals.get('notice_period_days', 90)
    
    if resp_rate >= 0.7:
        score *= 1.1
        score_breakdown.append(("Behavioral", 0.0, f"Response rate {resp_rate:.0%}"))
    elif resp_rate < 0.3:
        score *= 0.7
        score_breakdown.append(("Behavioral", 0.0, f"Low response rate {resp_rate:.0%}"))
    else:
        score *= 1.0
    
    if last_active:
        try:
            ref = datetime.strptime(last_active, "%Y-%m-%d")
            days_inactive = (datetime(2026, 6, 26) - ref).days
            if days_inactive > 180:
                score *= 0.7
                score_breakdown.append(("Behavioral", 0.0, f"Stale profile ({days_inactive}d inactive)"))
            elif days_inactive <= 30:
                score *= 1.0
        except:
            pass
    
    if not open_to_work:
        score *= 0.8
    
    if notice <= 30:
        score *= 1.05
        score_breakdown.append(("Behavioral", 0.0, f"Sub-30d notice ({notice}d)"))
    elif notice > 120:
        score *= 0.95
    
    # Location bonus
    loc_lower = location.lower()
    if any(l in loc_lower for l in ['pune', 'noida', 'hyderabad', 'mumbai', 'delhi', 'bangalore']):
        score *= 1.05
        score_breakdown.append(("Location", 0.0, f"Preferred location: {location}"))
    elif signals.get('willing_to_relocate', False):
        score *= 1.02
        score_breakdown.append(("Location", 0.0, "Willing to relocate to India"))
    
    # Title-chasing penalty
    if len(career) >= 3:
        avg_tenure = sum(r.get('duration_months', 0) for r in career) / len(career)
        unique_companies = len(set(r.get('company', '') for r in career))
        if unique_companies >= 3 and avg_tenure < 18:
            score *= 0.9
            score_breakdown.append(("Career", 0.0, f"Frequent job changes (avg {avg_tenure:.0f} months)"))
    
    # Clamp score
    score = max(0.0, min(1.0, score))
    
    # ===== GENERATE REASONING =====
    reasoning_parts = []
    
    # Title and years
    if 5 <= years <= 9:
        reasoning_parts.append(f"{title} ({int(years)}yrs)")
    elif years > 9:
        reasoning_parts.append(f"{title} ({int(years)}yrs, senior)")
    else:
        reasoning_parts.append(f"{title} ({int(years)}yrs)")
    
    # Core evidence
    if career_details['has_search_ranking_production']:
        evidence_str = '; '.join(career_details['search_ranking_evidence'][:1])
        reasoning_parts.append(f"built ranking/search at {evidence_str}")
    
    if career_details['has_embeddings_production']:
        evidence_str = '; '.join(career_details['embedding_evidence'][:1])
        reasoning_parts.append(f"deployed embeddings ({evidence_str})")
    
    if career_details['has_eval_framework']:
        evidence_str = '; '.join(career_details['eval_evidence'][:1])
        reasoning_parts.append(f"eval framework ({evidence_str})")
    
    if career_details['has_vector_db']:
        reasoning_parts.append("vector DB experience")
    
    if career_details['has_llm']:
        reasoning_parts.append("LLM integration")
    
    # Behavioral
    if resp_rate >= 0.7:
        reasoning_parts.append(f"response {resp_rate:.0%}")
    elif resp_rate >= 0.5:
        reasoning_parts.append(f"response {resp_rate:.0%}")
    
    if open_to_work:
        if notice <= 30:
            reasoning_parts.append(f"available ({notice}d notice)")
        elif notice <= 60:
            reasoning_parts.append(f"available ({notice}d notice)")
        else:
            reasoning_parts.append("open to work")
    
    if any(l in loc_lower for l in ['pune', 'noida', 'hyderabad', 'mumbai', 'delhi', 'bangalore']):
        reasoning_parts.append("India-based")
    
    reasoning = "; ".join(reasoning_parts[:6])
    
    # Cap reasoning length
    if len(reasoning) > 200:
        reasoning = reasoning[:197] + "..."
    
    return score, reasoning, {
        'evidence': evidence,
        'career_details': career_details,
        'skill_details': skill_details,
        'score_breakdown': score_breakdown,
        'product_ratio': product_months / total_months if total_months > 0 else 0,
        'years': years,
        'resp_rate': resp_rate,
        'open_to_work': open_to_work,
        'notice': notice,
    }


# ============================================================
# Main Pipeline
# ============================================================
def main():
    start = time.time()
    
    print("Loading candidates...")
    candidates = []
    with open(CANDIDATES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line.strip()))
    
    print(f"Loaded {len(candidates)} candidates")
    
    print("Computing JD-compliant scores with evidence extraction...")
    results = []
    disqualified = 0
    
    for candidate in candidates:
        cid = candidate.get('candidate_id', '')
        
        # Check disqualifiers FIRST
        is_disqualified, reason = check_disqualifiers(candidate)
        if is_disqualified:
            disqualified += 1
            continue
        
        # Compute score
        score, reasoning, evidence = compute_score(candidate)
        
        results.append({
            'candidate_id': cid,
            'score': score,
            'reasoning': reasoning,
            'evidence': evidence
        })
    
    print(f"Disqualified {disqualified} candidates")
    print(f"Qualified candidates: {len(results)}")
    
    # Sort by score descending, then candidate_id for tie-breaking
    results.sort(key=lambda x: (-x['score'], x['candidate_id']))
    
    # Take top 100
    top100 = results[:100]
    
    # Write submission
    print("Writing submission...")
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['candidate_id', 'rank', 'score', 'reasoning'])
        writer.writeheader()
        for rank, item in enumerate(top100, 1):
            writer.writerow({
                'candidate_id': item['candidate_id'],
                'rank': rank,
                'score': item['score'],
                'reasoning': item['reasoning']
            })
    
    # Statistics
    print("\n" + "="*60)
    print("RANKING STATISTICS")
    print("="*60)
    print(f"Total candidates processed: {len(candidates)}")
    print(f"Disqualified: {disqualified}")
    print(f"Qualified: {len(results)}")
    print(f"Top 100 score range: {top100[0]['score']:.4f} - {top100[-1]['score']:.4f}")
    
    # Match level statistics
    fully_match = 0
    partial_match = 0
    weak_match = 0
    
    for item in top100:
        score = item['score']
        if score >= 0.75:
            fully_match += 1
        elif score >= 0.60:
            partial_match += 1
        else:
            weak_match += 1
    
    print(f"\nFully Match (score >= 0.75): {fully_match}")
    print(f"Partial Match (0.60 <= score < 0.75): {partial_match}")
    print(f"Weak Match (score < 0.60): {weak_match}")
    
    reliable = fully_match + partial_match
    reliability_pct = (reliable / min(100, len(top100))) * 100 if top100 else 0
    print(f"\nReliable Fit Percentage: {reliability_pct:.1f}%")
    
    # Check for false positives
    false_positives = []
    for item in top100:
        ev = item['evidence']
        cd = ev['career_details']
        if not cd['has_search_ranking_production'] and not cd['has_embeddings_production']:
            false_positives.append(item['candidate_id'])
    
    print(f"Potential false positives (no search/ranking/embeddings production): {len(false_positives)}")
    if false_positives:
        print(f"  IDs: {', '.join(false_positives[:10])}")
    
    # Check for ranking order issues
    order_issues = []
    for i in range(len(top100) - 1):
        if top100[i]['score'] < top100[i+1]['score']:
            order_issues.append((top100[i]['candidate_id'], top100[i+1]['candidate_id']))
    
    print(f"Ranking order issues: {len(order_issues)}")
    
    print(f"\nTime elapsed: {time.time() - start:.1f}s")
    print(f"\nOutput written to: {OUTPUT_FILE}")
    
    # Confidence score
    confidence = reliability_pct * (1 - len(false_positives)/100) * (1 - len(order_issues)/100)
    print(f"\nFinal Confidence Score: {confidence:.1f}/100")


if __name__ == '__main__':
    main()
