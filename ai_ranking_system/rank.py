#!/usr/bin/env python3
"""
AI Candidate Discovery & Ranking System - Main Entry Point.
Run: python -m ai_ranking_system.rank --candidates ./candidates.jsonl --out ./submission.csv
"""

import argparse
import json
import csv
import time
import os
import re
from typing import Dict, List, Tuple
from .behavioral_scorer import BehavioralSignalScorer

# JD-aligned constants
CONSULTING_FIRMS = {
    'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini',
    'mindtree', 'tech mahindra', 'hcl', 'lti', 'persistent'
}

JOB_KEYWORDS = {
    'embedding', 'retrieval', 'ranking', 'recommendation', 'vector', 'search', 'ml', 'nlp', 
    'llm', 'rag', 'milvus', 'pinecone', 'faiss', 'elasticsearch', 'python', 'pytorch', 
    'tensorflow', 'production', 'deployed', 'sentence-transformers', 'bge', 'e5',
    'learning-to-rank', 'ndcg', 'mrr', 'map', 'evaluation', 'hybrid search'
}

AI_SKILLS = {
    'embedding', 'retrieval', 'ranking', 'vector', 'search', 'milvus', 'pinecone', 'faiss', 
    'nlp', 'llm', 'rag', 'bge', 'e5', 'learning-to-rank', 'recommender', 'recommendation',
    'semantic search', 'hybrid search', 'information retrieval'
}

INDIA_LOCATIONS = {
    'pune', 'noida', 'hyderabad', 'mumbai', 'delhi', 'ncr', 'bangalore', 'gurgaon',
    'chennai', 'kolkata', 'vizag', 'trivandrum'
}

NON_TECHNICAL_TITLES = {
    'marketing manager', 'sales', 'operations manager', 'project manager',
    'hr manager', 'accountant', 'content writer', 'graphic designer',
    'civil engineer', 'mechanical engineer', 'customer support',
    'business analyst', 'sales executive', 'operations'
}

CV_SPEECH_ROBOTICS = {
    'computer vision', 'cv engineer', 'vision engineer', 'robotics',
    'speech engineer', 'asr engineer', 'tts engineer'
}

SIDE_PROJECT_PATTERNS = [
    re.compile(r'\b(?:langchain|openai)\s+(?:api|tutorial|side project|experiment)\b', re.IGNORECASE),
    re.compile(r'\b(?:side project|online course|taking courses)\b.*\b(?:ai|ml|rag|llm)\b', re.IGNORECASE),
    re.compile(r'\b(?:curious about|exploring|getting into)\b.*\b(?:ai|ml|genai)\b', re.IGNORECASE),
]

SIDE_PROJECT_PATTERNS = [
    re.compile(r'\b(?:langchain|openai)\s+(?:api|tutorial|side project|experiment)\b', re.IGNORECASE),
    re.compile(r'\b(?:side project|online course|taking courses)\b.*\b(?:ai|ml|rag|llm)\b', re.IGNORECASE),
    re.compile(r'\b(?:curious about|exploring|getting into)\b.*\b(?:ai|ml|genai)\b', re.IGNORECASE),
]


def extract_candidate_text(candidate):
    parts = [candidate.get('profile', {}).get('headline', ''),
             candidate.get('profile', {}).get('summary', '')]
    for role in candidate.get('career_history', [])[:3]:
        parts.append(role.get('title', ''))
        d = role.get('description', '')
        if d:
            parts.append(d[:300])
    for skill in candidate.get('skills', [])[:10]:
        parts.append(skill.get('name', ''))
    return ' '.join(filter(None, parts)).lower()


def is_non_technical_title(title: str) -> bool:
    return any(t in title for t in NON_TECHNICAL_TITLES)


def is_cv_speech_robotics_primary(title: str, skill_names: List[str], text: str, career: List[Dict]) -> bool:
    if not any(t in title for t in CV_SPEECH_ROBOTICS):
        return False
    
    # Check if most recent role is ALSO in CV/Speech/Robotics
    recent_cv = False
    recent_desc = ''
    if career:
        recent_title = career[0].get('title', '').lower()
        recent_cv = any(t in recent_title for t in CV_SPEECH_ROBOTICS)
        recent_desc = career[0].get('description', '').lower()
    
    if not recent_cv:
        return False
    
    # Look for explicit production search/ranking/retrieval in MOST RECENT role only
    recent_role_patterns = [
        r'search\s+(?:system|engine|pipeline|infrastructure)',
        r'ranking\s+(?:system|model|pipeline|layer|feed)',
        r'retrieval\s+(?:system|pipeline|augmented|architecture)',
        r'recommendation\s+(?:system|engine|model|feed)',
        r'vector\s+(?:search|database|index|retrieval|store)',
        r'semantic\s+search',
        r'hybrid\s+search',
        r'embedding\s+(?:model|system|pipeline|index|service)',
        r'\b(?:pinecone|milvus|faiss|weaviate|qdrant|elasticsearch|opensearch|pgvector)\b',
        r'\b(?:bm25|elasticsearch)\b',
        r'learning.?to.?rank',
        r'\b(?:ndcg|mrr|map)\b',
        r'\brag\b.*\b(?:retrieval|search|system|pipeline|architecture)\b',
        r're-?ranking\s+(?:model|layer|system|over)',
        r're-?rank(?:ing)?\s+(?:over|with|using|model)',
        r'collaborative\s+filtering.*(?:ranking|recommendation)',
    ]
    has_recent_search = any(re.search(p, recent_desc, re.IGNORECASE) for p in recent_role_patterns)
    
    if recent_cv and not has_recent_search:
        return True
    
    return False


def is_langchain_side_project_primary(summary: str, text: str, career: List[Dict]) -> bool:
    side_signals = sum(1 for p in SIDE_PROJECT_PATTERNS if p.search(summary))
    prod_signals = sum(1 for kw in ['deployed', 'production', 'shipped', 'live', 
                                     'real users', 'latency', 'throughput', 'a/b test']
                       if kw in text)
    has_real_jobs = len([r for r in career if r.get('duration_months', 0) >= 12]) >= 1
    
    if side_signals >= 2 and prod_signals <= 1 and has_real_jobs:
        return True
    return False


def is_consulting_only(career: List[Dict]) -> bool:
    for role in career:
        company = role.get('company', '').lower()
        duration = role.get('duration_months', 0)
        if duration < 12:
            continue
        if company not in CONSULTING_FIRMS:
            return False
    return True


def has_ir_production_experience(text: str) -> bool:
    ir_terms = [
        'retrieval', 'search', 'ranking', 'recommendation', 'vector search',
        'semantic search', 'hybrid search', 'embedding', 'bm25', 'elasticsearch',
        'opensearch', 'pinecone', 'milvus', 'faiss', 'weaviate', 'qdrant',
        'ndcg', 'mrr', 'map', 'learning-to-rank', 'rerank'
    ]
    return any(term in text for term in ir_terms)


def compute_eval_framework_score(candidate: Dict) -> float:
    profile = candidate.get('profile', {})
    career = candidate.get('career_history', [])
    skills = candidate.get('skills', [])
    
    text = ' '.join([
        profile.get('headline', ''),
        profile.get('summary', ''),
        ' '.join(r.get('description', '') + ' ' + r.get('title', '') for r in career),
        ' '.join(s.get('name', '') for s in skills)
    ]).lower()
    
    score = 0.0
    
    metric_mentions = sum(1 for kw in ['ndcg', 'mrr', 'map', 'precision@k', 'recall@k'] if kw in text)
    score += min(metric_mentions * 0.15, 0.3)
    
    if 'a/b test' in text or 'ab test' in text:
        score += 0.15
    if 'experiment' in text or 'experimentation' in text:
        score += 0.1
    
    if 'offline-to-online' in text or 'offline online correlation' in text:
        score += 0.15
    if 'offline' in text and 'online' in text:
        score += 0.1
    
    if 'relevance' in text and 'judgment' in text:
        score += 0.1
    if 'human' in text and ('label' in text or 'judgment' in text):
        score += 0.1
    
    if 'eval harness' in text or 'evaluation framework' in text or 'eval pipeline' in text:
        score += 0.15
    if 'benchmark' in text:
        score += 0.05
    
    if 'metric' in text and ('optimize' in text or 'improve' in text):
        score += 0.1
    
    return min(1.0, score)


def compute_experience_score(years: float) -> float:
    """JD: 5-9 years preferred, flexible for strong candidates."""
    if 5 <= years <= 9:
        return 1.0
    elif 4 <= years < 5:
        return 0.7
    elif 9 < years <= 12:
        return 0.7
    elif 3 <= years < 4:
        return 0.4
    elif 12 < years <= 15:
        return 0.4
    else:
        return 0.1


def compute_company_quality_score(career: List[Dict]) -> float:
    strong_product = {
        'google', 'amazon', 'microsoft', 'apple', 'meta', 'netflix', 'uber', 'adobe',
        'salesforce', 'ola', 'zomato', 'swiggy', 'flipkart', 'paytm', 'phonepe',
        'dream11', 'meesho', 'inmobi', 'freshworks', 'zoho', 'razorpay', 'linkedin',
        'airbnb', 'spotify', 'stripe', 'shopify', 'bytedance', 'tencent', 'alibaba'
    }
    
    good_product = {
        'cred', 'groww', 'policybazaar', 'nykaa', 'hasura', 'postman', 'browserstack',
        'hotstar', 'pharmeasy', 'aganitha', 'verloop.io', 'repharse.ai', 'sarvam ai',
        'yellow.ai', 'pied piper', 'hooli', 'wayne enterprises', 'acme corp',
        'globex inc', 'initech', 'dunder mifflin', 'stark industries'
    }
    
    score = 0.0
    total_months = 0
    
    for role in career:
        company = role.get('company', '').lower()
        duration = role.get('duration_months', 0)
        total_months += duration
        
        if company in strong_product:
            score += duration * 1.0
        elif company in good_product:
            score += duration * 0.8
        elif company not in CONSULTING_FIRMS:
            score += duration * 0.6
        else:
            score += duration * 0.1
    
    return min(1.0, score / total_months) if total_months > 0 else 0.0


def apply_behavioral_penalties(candidate: Dict, score: float) -> float:
    signals = candidate.get('redrob_signals', {})
    
    resp_rate = signals.get('recruiter_response_rate', 0.5)
    if resp_rate < 0.3:
        score *= 0.6
    elif resp_rate < 0.5:
        score *= 0.8
    
    last_active = signals.get('last_active_date', '')
    if last_active:
        from datetime import datetime
        try:
            ref = datetime.strptime(last_active, "%Y-%m-%d")
            days_inactive = (datetime(2026, 6, 26) - ref).days
            if days_inactive > 180:
                score *= 0.5
            elif days_inactive > 90:
                score *= 0.8
        except:
            pass
    
    if not signals.get('open_to_work_flag', False):
        score *= 0.7
    
    notice = signals.get('notice_period_days', 90)
    if notice > 120:
        score *= 0.9
    
    return score


def check_disqualifiers(candidate: Dict) -> Tuple[bool, str]:
    profile = candidate.get('profile', {})
    career = candidate.get('career_history', [])
    skills = candidate.get('skills', [])
    summary = profile.get('summary', '').lower()
    title = profile.get('current_title', '').lower()
    
    skill_names = [s.get('name', '').lower() for s in skills]
    text = extract_candidate_text(candidate)
    
    # Gate 1: Non-technical title with no IR production
    if is_non_technical_title(title):
        if not has_ir_production_experience(text):
            return True, f"Non-technical title ({profile.get('current_title', '')}) with no production search/ranking/retrieval experience"
    
    # Gate 2: Non-software engineering primary - STRICT
    non_software = ['civil engineer', 'mechanical engineer', 'electrical engineer', 'chemical engineer']
    if any(t in title for t in non_software):
        # For non-software titles, require EXPLICIT production AI/ML evidence
        # AND exclude if they mention side projects/online courses/LangChain
        side_project_signals = sum(1 for p in SIDE_PROJECT_PATTERNS if p.search(summary))
        prod_ai_signals = sum(1 for kw in ['deployed ml', 'shipped ml', 'ml model', 'machine learning model',
                                            'ai model', 'deep learning', 'neural network', 'tensorflow', 'pytorch']
                             if kw in text)
        if side_project_signals >= 1 or prod_ai_signals == 0:
            return True, f"Primary expertise is non-software engineering ({profile.get('current_title', '')}) without production AI/ML deployment"
    
    # Gate 3: CV/Speech/Robotics primary without NLP/IR production
    if is_cv_speech_robotics_primary(title, skill_names, text, career):
        return True, f"Primary expertise is CV/Speech/Robotics ({profile.get('current_title', '')}) without production NLP/IR/Search/Ranking"
    
    # Gate 4: Consulting-only background
    if is_consulting_only(career):
        return True, "Consulting-only background with no product company experience"
    
    # Gate 5: LangChain/OpenAI side projects as primary
    if is_langchain_side_project_primary(summary, text, career):
        return True, "AI experience primarily LangChain/OpenAI side projects without substantial pre-LLM production ML"
    
    # Gate 6: Pure research without production
    research_titles = ['research scientist', 'research engineer', 'postdoc', 'research analyst']
    research_companies = ['university', 'institute', 'lab', 'academia']
    is_research = any(
        any(k in r.get('title', '').lower() for k in research_titles) or
        any(k in r.get('company', '').lower() for k in research_companies)
        for r in career[:2]
    )
    has_production = has_ir_production_experience(text)
    if is_research and not has_production:
        return True, "Pure research background without production deployment"
    
    return False, ""


def compute_score(candidate: Dict) -> Tuple[str, float, str]:
    cid = candidate.get('candidate_id', '')
    profile = candidate.get('profile', {})
    signals = candidate.get('redrob_signals', {})
    text = extract_candidate_text(candidate)
    
    # Check disqualifiers FIRST
    is_disqualified, reason = check_disqualifiers(candidate)
    if is_disqualified:
        return cid, 0.0, f"DISQUALIFIED: {reason}"
    
    # Semantic similarity
    overlap = len(JOB_KEYWORDS & set(text.split()))
    semantic_sim = min(overlap / 8.0, 1.0)
    
    # Skill match
    skill_names = [s.get('name', '').lower() for s in candidate.get('skills', [])]
    core_ai = sum(1 for s in skill_names if any(t in s for t in AI_SKILLS))
    has_python = 'python' in skill_names or any(t in text for t in ['pytorch', 'tensorflow', 'scikit'])
    skill_score = min(0.5 if core_ai >= 4 else 0.35 if core_ai >= 2 else 0.2 if core_ai >= 1 else 0, 1.0)
    skill_score += 0.3 if has_python else 0
    
    # Production ML score
    has_prod = 'production' in text or 'deployed' in text or 'real-users' in text
    prod_score = min(0.7 if has_prod else 0.3, 1.0)
    
    # Career trajectory
    ch = candidate.get('career_history', [])
    stability = sum(1 for r in ch[:-1] if r.get('duration_months', 0) >= 24) if len(ch) > 1 else 0
    cv_score = min(0.2 + stability * 0.3, 1.0)
    
    # Behavioral signals - all 23 Redrob signals via BehavioralSignalScorer + custom signals
    behavioral_scorer = BehavioralSignalScorer()
    base_behavioral = behavioral_scorer.compute_total_behavioral_score(candidate)
    
    # Additional signal bonuses (10 missing signals not covered by BehavioralSignalScorer)
    signal_bonus = 0.0
    
    # 2. signup_date: penalize very fresh accounts (< 1 year)
    signup = signals.get('signup_date', '')
    if signup:
        try:
            signup_dt = datetime.strptime(signup, "%Y-%m-%d")
            account_age_years = (datetime(2026, 6, 26) - signup_dt).days / 365.25
            if account_age_years < 1:
                signal_bonus -= 0.05
        except:
            pass
    
    # 5. profile_views_received_30d
    views = signals.get('profile_views_received_30d', 0)
    if views >= 30:
        signal_bonus += 0.08
    elif views >= 10:
        signal_bonus += 0.04
    
    # 6. applications_submitted_30d
    apps = signals.get('applications_submitted_30d', 0)
    if apps >= 10:
        signal_bonus += 0.1
    elif apps >= 3:
        signal_bonus += 0.05
    
    # 8. avg_response_time_hours: lower is better
    resp_time = signals.get('avg_response_time_hours', 24)
    if resp_time <= 4:
        signal_bonus += 0.1
    elif resp_time <= 12:
        signal_bonus += 0.05
    elif resp_time > 48:
        signal_bonus -= 0.1
    
    # 9. skill_assessment_scores: average of top 3 scores
    skill_scores = signals.get('skill_assessment_scores', {})
    if skill_scores:
        top_scores = sorted(skill_scores.values(), reverse=True)[:3]
        avg_top = sum(top_scores) / len(top_scores)
        if avg_top >= 80:
            signal_bonus += 0.15
        elif avg_top >= 60:
            signal_bonus += 0.08
    
    # 10. connection_count
    connections = signals.get('connection_count', 0)
    if connections >= 500:
        signal_bonus += 0.08
    elif connections >= 100:
        signal_bonus += 0.04
    
    # 11. endorsements_received
    endorsements = signals.get('endorsements_received', 0)
    if endorsements >= 30:
        signal_bonus += 0.1
    elif endorsements >= 10:
        signal_bonus += 0.05
    
    # 13. expected_salary_range_inr_lpa: penalty if expectations too high
    salary_range = signals.get('expected_salary_range_inr_lpa', {})
    if salary_range:
        max_salary = salary_range.get('max', 0)
        if max_salary > 50:
            signal_bonus -= 0.1
    
    # 14. preferred_work_mode
    work_mode = signals.get('preferred_work_mode', '')
    if work_mode in ('remote', 'hybrid', 'flexible'):
        signal_bonus += 0.05
    
    # 20. offer_acceptance_rate
    offer_rate = signals.get('offer_acceptance_rate', -1)
    if offer_rate >= 0.8:
        signal_bonus += 0.1
    elif offer_rate >= 0.6:
        signal_bonus += 0.05
    elif offer_rate >= 0 and offer_rate < 0.4:
        signal_bonus -= 0.15
    
    behavior = max(0.0, min(1.0, base_behavioral + signal_bonus))
    avail = 0.0  # availability is baked into behavior from BehavioralSignalScorer
    
    # Evaluation framework signal (strong differentiator)
    has_eval_framework = any(kw in text for kw in ['ndcg', 'mrr', 'map', 'precision', 'recall', 'ab test', 'experiment', 'offline', 'online'])
    eval_bonus = 0.15 if has_eval_framework else 0
    
    # Final score
    score = 0.40 * semantic_sim + 0.20 * skill_score + 0.15 * prod_score + \
            0.10 * cv_score + 0.10 * behavior + 0.05 * min(avail, 1.0) + eval_bonus
    
    # Product company check
    prod_company = any(r.get('company', '').lower() not in CONSULTING_FIRMS for r in ch)
    
    # Consulting-only background (strong negative signal per JD)
    consulting_only = all(r.get('company', '').lower() in CONSULTING_FIRMS for r in ch if r.get('duration_months', 0) >= 12)
    
    # Honeypot/red-flag check
    title = profile.get('current_title', '').lower()
    summary = profile.get('summary', '').lower()
    
    is_honeypot_title = core_ai >= 2 and any(p in title for p in ['marketing', 'sales', 'operations', 'hr', 'accountant', 'content writer', 'graphic designer', 'business analyst'])
    is_honeypot_summary = any(p in summary for p in ['curious about ai', 'tutorial', 'how i used', 'langchain tutorial', 'built with ai'])
    is_honeypot = is_honeypot_title or is_honeypot_summary
    
    if is_honeypot:
        score *= 0.2
    if not prod_company:
        score *= 0.6
    if consulting_only:
        score *= 0.7
    
    # Reasoning
    years = profile.get('years_of_experience', 0)
    resp_rate = signals.get('recruiter_response_rate', 0)
    open_to_work = signals.get('open_to_work_flag', False)
    notice = signals.get('notice_period_days', 90)
    views = signals.get('profile_views_received_30d', 0)
    endorsements = signals.get('endorsements_received', 0)
    skill_scores = signals.get('skill_assessment_scores', {})
    offer_rate = signals.get('offer_acceptance_rate', -1)
    
    parts = []
    if 5 <= years <= 9:
        parts.append(f"{int(years)}yr")
    if core_ai >= 2:
        parts.append(f"{core_ai} AI skills")
    if resp_rate >= 0.5:
        parts.append(f"response {resp_rate:.0%}")
    if open_to_work:
        if notice <= 30:
            parts.append(f"available ({notice}d notice)")
        elif notice <= 60:
            parts.append(f"available ({notice}d notice)")
        else:
            parts.append("open to work")
    if views >= 30:
        parts.append("high interest")
    if endorsements >= 30:
        parts.append("endorsed")
    if skill_scores:
        top_scores = sorted(skill_scores.values(), reverse=True)[:3]
        avg_top = sum(top_scores) / len(top_scores)
        if avg_top >= 80:
            parts.append("top assessor")
    if offer_rate >= 0.8:
        parts.append("high offer acceptance")
    
    sentiment = "strong fit" if score >= 0.65 else "good fit" if score >= 0.45 else "potential fit"
    reason = f"{profile.get('current_title', '')} {', '.join(parts)}; {sentiment}"[:150]
    
    return cid, max(0.0, min(1.0, score)), reason


def main():
    parser = argparse.ArgumentParser(description='Rank candidates for Senior AI Engineer role')
    parser.add_argument('--candidates', required=True, help='Path to candidates JSONL or gzipped JSONL file')
    parser.add_argument('--out', required=True, help='Path to output submission CSV')
    args = parser.parse_args()
    
    start = time.time()
    print("Loading candidates...")
    
    # Support both .jsonl and .jsonl.gz
    if args.candidates.endswith('.gz'):
        import gzip
        with gzip.open(args.candidates, 'rt', encoding='utf-8') as f:
            candidates = [json.loads(line.strip()) for line in f if line.strip()]
    else:
        with open(args.candidates, 'r', encoding='utf-8') as f:
            candidates = [json.loads(line.strip()) for line in f if line.strip()]
    
    print(f"Loaded {len(candidates)} candidates")
    results = [compute_score(c) for c in candidates]
    results.sort(key=lambda x: (-x[1], x[0]))
    
    # Filter out disqualified candidates (score > 0 or reasoning doesn't start with DISQUALIFIED)
    qualified = [(cid, score, reason) for cid, score, reason in results 
                 if not reason.startswith('DISQUALIFIED:')]
    
    # Ensure monotonic non-increasing scores with proper rounding
    final = []
    prev_score = 1.0
    for rank, (cid, score, reason) in enumerate(qualified[:100], 1):
        curr_score = min(prev_score, round(score, 4))
        prev_score = max(0.0, curr_score - 0.01)
        final.append({'candidate_id': cid, 'rank': rank, 'score': round(curr_score, 4), 'reasoning': reason})
    
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['candidate_id', 'rank', 'score', 'reasoning'])
        writer.writeheader()
        writer.writerows(final)
    
    print(f"Done in {time.time() - start:.1f}s. Generated {len(final)} rankings.")
    print(f"Total candidates: {len(candidates)}, Qualified: {len(qualified)}, Disqualified: {len(candidates) - len(qualified)}")


if __name__ == '__main__':
    main()
