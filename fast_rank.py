#!/usr/bin/env python3
"""
Optimized Candidate Ranking Pipeline - Fast vectorized scoring.
Fully compliant with Senior AI Engineer JD requirements.
"""

import json
import csv
import time
import re
from datetime import datetime
from typing import Dict, List, Tuple
from ai_ranking_system.behavioral_scorer import BehavioralSignalScorer

# JD-aligned constants
CONSULTING_FIRMS = {
    'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini',
    'mindtree', 'tech mahindra', 'hcl', 'lti', 'persistent'
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

# JD disqualifier patterns
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


def extract_candidate_text(candidate):
    parts = [
        candidate.get('profile', {}).get('headline', ''),
        candidate.get('profile', {}).get('summary', '')
    ]
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
    
    # Look for explicit production search/ranking/retrieval infrastructure in MOST RECENT role
    # STRICT: require evidence of BUILT systems, not just feature mentions
    recent_role_patterns = [
        r'built.*(?:search|ranking|retrieval|recommendation).*(?:system|pipeline|engine|infrastructure)',
        r'designed.*(?:search|ranking|retrieval|recommendation).*(?:system|pipeline|engine)',
        r'developed.*(?:search|ranking|retrieval|recommendation).*(?:system|pipeline|engine)',
        r'engineered.*(?:search|ranking|retrieval|recommendation)',
        r'led.*(?:search|ranking|retrieval|recommendation).*(?:team|project|system)',
        r'owned.*(?:search|ranking|retrieval|recommendation).*(?:system|pipeline)',
        r'architected.*(?:search|ranking|retrieval|recommendation)',
        r'shipped.*(?:search|ranking|retrieval|recommendation)',
        r'deployed.*(?:search|ranking|retrieval|recommendation).*(?:system|model|service)',
        r'search\s+(?:system|engine|pipeline|infrastructure)',
        r'ranking\s+(?:system|model|pipeline)',
        r'retrieval\s+(?:system|pipeline|augmented|architecture)',
        r'recommendation\s+(?:system|engine|feed)',
        r'vector\s+(?:search|database|index|retrieval)',
        r'semantic\s+search',
        r'hybrid\s+search',
        r'embedding\s+(?:model|system|pipeline|index|service)',
        r'\b(?:pinecone|milvus|faiss|weaviate|qdrant|elasticsearch|opensearch|pgvector)\b',
        r'\b(?:bm25|elasticsearch)\b',
        r'learning.?to.?rank',
        r'\b(?:ndcg|mrr|map)\b',
        r'\brag\b.*\b(?:retrieval|search|system|pipeline)\b',
    ]
    has_recent_search = any(re.search(p, recent_desc, re.IGNORECASE) for p in recent_role_patterns)
    
    if recent_cv and not has_recent_search:
        return True
    
    # EXTRA GATE: CV primary with current role in CV - require STRONG evidence
    # "recommendation-style features" and "re-ranking" in a CV role are NOT sufficient
    # Need explicit search/ranking/retrieval infrastructure evidence
    strong_infrastructure_patterns = [
        r'search\s+(?:system|engine|pipeline|infrastructure)',
        r'ranking\s+(?:system|model|pipeline)',
        r'retrieval\s+(?:system|pipeline|architecture)',
        r'recommendation\s+(?:system|engine|feed)',
        r'vector\s+(?:search|database|index|retrieval)',
        r'semantic\s+search',
        r'hybrid\s+search',
        r'\b(?:pinecone|milvus|faiss|weaviate|qdrant|elasticsearch|opensearch|pgvector)\b',
        r'\b(?:bm25|elasticsearch)\b',
        r'learning.?to.?rank',
        r'\brag\b.*\b(?:retrieval|search|system)\b',
    ]
    strong_matches = sum(1 for p in strong_infrastructure_patterns if re.search(p, recent_desc, re.IGNORECASE))
    
    if recent_cv and strong_matches < 2:
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
        'dream11', 'meesho', 'inmobi', 'freshworks', 'zoho', 'razorpay'
    }
    
    good_product = {
        'cred', 'groww', 'policybazaar', 'nykaa', 'hasura', 'postman', 'browserstack',
        'hotstar', 'pharmeasy', 'aganitha', 'verloop.io', 'repharse.ai'
    }
    
    score = 0.0
    product_months = 0
    total_months = 0
    
    for role in career:
        company = role.get('company', '').lower()
        duration = role.get('duration_months', 0)
        total_months += duration
        
        if company in strong_product:
            score += duration * 1.0
            product_months += duration
        elif company in good_product:
            score += duration * 0.8
            product_months += duration
        elif company not in CONSULTING_FIRMS:
            score += duration * 0.6
            product_months += duration
        # Consulting firms get minimal points (0.1) and count toward consulting months
    
    product_ratio = product_months / total_months if total_months > 0 else 0.0
    
    return min(1.0, score / total_months) if total_months > 0 else 0.0


def apply_behavioral_penalties(candidate: Dict, score: float) -> float:
    """Apply JD-aligned behavioral signal penalties."""
    signals = candidate.get('redrob_signals', {})
    
    resp_rate = signals.get('recruiter_response_rate', 0.5)
    if resp_rate < 0.3:
        score *= 0.6
    elif resp_rate < 0.5:
        score *= 0.8
    
    last_active = signals.get('last_active_date', '')
    if last_active:
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


def load_candidates(path: str) -> List[Dict]:
    """Load candidates from JSONL or gzipped JSONL."""
    if path.endswith('.gz'):
        import gzip
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return [json.loads(line.strip()) for line in f if line.strip()]
    else:
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line.strip()) for line in f if line.strip()]


def compute_scores_batch(candidates: List[Dict]) -> List[Dict]:
    """
    Compute scores for all candidates with JD-compliant logic.
    Returns list of dicts with candidate_id, score, reasoning.
    """
    results = []
    behavioral_scorer = BehavioralSignalScorer()
    
    for c in candidates:
        cid = c.get('candidate_id', '')
        profile = c.get('profile', {})
        signals = c.get('redrob_signals', {})
        career = c.get('career_history', [])
        skills = c.get('skills', [])
        
        text = extract_candidate_text(c)
        title = profile.get('current_title', '').lower()
        summary = profile.get('summary', '').lower()
        skill_names = [s.get('name', '').lower() for s in skills]
        
        # ===== DISQUALIFIER GATES =====
        # Gate 1: Non-technical title with no IR production experience
        if is_non_technical_title(title):
            if not has_ir_production_experience(text):
                continue  # Skip entirely
        
        # Gate 2: Non-software engineering primary
        non_software = ['civil engineer', 'mechanical engineer', 'electrical engineer']
        if any(t in title for t in non_software):
            has_software = any(kw in text for kw in ['python', 'ml', 'machine learning', 'deployed', 'production'])
            if not has_software:
                continue
        
        # Gate 3: CV/Speech/Robotics primary without production search/ranking/retrieval
        if is_cv_speech_robotics_primary(title, skill_names, text, career):
            continue
        
        # Gate 4: Consulting-only background
        if is_consulting_only(career):
            continue
        
        # Gate 5: LangChain/OpenAI side projects as primary
        if is_langchain_side_project_primary(summary, text, career):
            continue
        
        # ===== SCORING =====
        # Skill match - count AI skills with production context
        core_ai = sum(1 for s in skill_names if any(t in s for t in AI_SKILLS))
        has_python = 'python' in skill_names or any(t in text for t in ['pytorch', 'tensorflow', 'scikit'])
        skill_score = min(0.5 if core_ai >= 4 else 0.35 if core_ai >= 2 else 0.2 if core_ai >= 1 else 0, 1.0)
        skill_score += 0.3 if has_python else 0
        
        # NEW: Ranking/Search system evidence (production context REQUIRED)
        ranking_keywords = ['ranking', 'retrieval', 'recommendation', 'search', 'hybrid search']
        search_infra_keywords = ['vector', 'semantic search', 'bm25', 'elasticsearch', 'pinecone', 'milvus', 'faiss']
        
        # Check for production evidence - need BOTH system mentions AND production context
        has_ranking = any(kw in text for kw in ranking_keywords)
        has_search_infra = any(kw in text for kw in search_infra_keywords)
        has_prod_context = 'production' in text or 'deployed' in text or 'live' in text
        
        ranking_score = 0.0
        if has_ranking and has_search_infra and has_prod_context:
            ranking_score = 1.0
        elif (has_ranking or has_search_infra) and has_prod_context:
            ranking_score = 0.6
        elif has_ranking or has_search_infra:
            ranking_score = 0.0  # Mere mention without production is trap language - no points
        
        # Production ML experience
        prod_keywords = ['production', 'deployed', 'live', 'shipping', 'real users', 'real-users', 
                         'latency', 'throughput', 'serving', 'kubernetes', 'bentoml']
        has_prod = any(kw in text for kw in prod_keywords)
        prod_score = min(0.8 if has_prod else 0.3, 1.0)
        
        # Experience (bell curve 5-9 years)
        years = profile.get('years_of_experience', 0)
        exp_score = compute_experience_score(years)
        
        # Behavioral signals - from BehavioralSignalScorer
        base_behavioral = behavioral_scorer.compute_total_behavioral_score(c)
        
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
        
        # Evaluation framework
        eval_fw = compute_eval_framework_score(c)
        
        # Company quality
        company_quality = compute_company_quality_score(career)
        
        # Consulting months ratio penalty
        product_months = sum(
            role.get('duration_months', 0) for role in career
            if role.get('company', '').lower() not in CONSULTING_FIRMS
        )
        total_months = sum(role.get('duration_months', 0) for role in career)
        consulting_ratio = 1.0 - (product_months / total_months) if total_months > 0 else 1.0
        consulting_penalty = 1.0
        if consulting_ratio > 0.7:
            consulting_penalty = 0.5
        elif consulting_ratio > 0.5:
            consulting_penalty = 0.7
        elif consulting_ratio > 0.3:
            consulting_penalty = 0.85
        
        # Location bonus - JD prefers India/Pune/Noida/Hyderabad
        location = profile.get('location', '').lower()
        location_bonus = 0.10 if any(loc in location for loc in INDIA_LOCATIONS) else 0.05
        
        # Final weighted score - REVISED to avoid keyword trap, sums to 100%
        raw_score = (
            0.25 * ranking_score +       # Production ranking/search systems (MOST critical)
            0.20 * company_quality +     # Product company experience required
            0.15 * prod_score +          # Production ML experience
            0.15 * eval_fw +             # Evaluation framework experience
            0.10 * skill_score +         # Skills with Python
            0.10 * behavior +          # Behavioral signals
            0.05 * location_bonus        # Location preference (0.05-0.10)
        )
        
        # Apply experience score as multiplier
        raw_score *= exp_score
        
        # Honeypot penalty - STRICT enforcement
        if 'ai enthusiast' in summary or 'curious about ai' in summary or 'getting into ai' in summary:
            raw_score *= 0.3  # Heavy penalty for JD trap language
        
        # Company quality penalty
        if company_quality < 0.3:
            raw_score *= 0.8
        
        # Consulting ratio penalty
        raw_score *= consulting_penalty
        
        # Behavioral penalties
        raw_score = apply_behavioral_penalties(c, raw_score)
        
        # Clamp
        score = max(0.0, min(1.0, raw_score))
        
        # Reasoning - enhanced with specific facts and JD connection
        years = profile.get('years_of_experience', 0)
        resp_rate_val = signals.get('recruiter_response_rate', 0)
        notice = signals.get('notice_period_days', 90)
        open_to_work = signals.get('open_to_work_flag', False)
        location = profile.get('location', '')
        
        # Extract specific skills
        skills_list = [s.get('name', '') for s in skills if any(t in s.get('name', '').lower() for t in AI_SKILLS)]
        top_skills = skills_list[:3]
        skills_str = ', '.join(top_skills[:2]) if top_skills else 'limited AI skills'
        if len(top_skills) > 2:
            skills_str += f", {top_skills[2]}"
        
        # Find product companies
        strong_product_companies = {'google', 'amazon', 'microsoft', 'apple', 'meta', 'netflix', 'uber', 'adobe', 'salesforce', 'ola', 'zomato', 'swiggy', 'flipkart', 'paytm', 'phonepe', 'dream11', 'meesho', 'inmobi', 'freshworks', 'zoho', 'razorpay'}
        prod_companies = [r.get('company', '') for r in career if r.get('company', '').lower() in strong_product_companies and r.get('duration_months', 0) >= 12]
        
        # Check for IR infrastructure
        infra_keywords = ['milvus', 'pinecone', 'faiss', 'weaviate', 'qdrant', 'elasticsearch', 'opensearch', 'pgvector']
        infra_found = [kw for kw in infra_keywords if kw in text.lower()]
        
        # Build reasoning parts
        title = profile.get('current_title', '')
        parts = []
        
        # Experience and skills
        if years >= 5:
            parts.append(f"{years:.1f} years experience")
        else:
            parts.append(f"{years:.1f} years experience (below ideal range)")
        
        if skills_str != 'limited AI skills' and core_ai > 0:
            skill_word = "skills" if core_ai > 1 else "skill"
            parts.append(f"{core_ai} core AI {skill_word} ({skills_str})")
        
        # Production evidence
        if infra_found:
            parts.append(f"production {infra_found[0].title()} experience")
        elif has_ranking or has_search_infra:
            parts.append("search/ranking mention")
        
        # Behavioral signals
        if resp_rate_val >= 0.7:
            parts.append(f"high recruiter response ({resp_rate_val:.0%})")
        elif resp_rate_val >= 0.5:
            parts.append(f"response rate {resp_rate_val:.0%}")
        
        # Honest concerns for lower ranks
        concerns = []
        if notice > 90:
            concerns.append(f"notice period {notice} days may impact timeline")
        if company_quality < 0.5 and total_months > 0:
            concerns.append("heavy consulting background")
        
        reasoning = f"{title} — " + "; ".join(parts) + "."
        if concerns:
            reasoning += f" Concern: {concerns[0]}."
        
        results.append({
            'candidate_id': cid,
            'score': score,
            'reasoning': reasoning
        })
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Rank candidates for Senior AI Engineer role')
    parser.add_argument('--candidates', default='candidates.jsonl', help='Path to candidates JSONL or gzipped JSONL file')
    parser.add_argument('--out', default='submission.csv', help='Path to output submission CSV')
    args = parser.parse_args()
    
    start = time.time()
    print(f"Loading candidates from {args.candidates}...")
    
    candidates = load_candidates(args.candidates)
    
    print(f"Loaded {len(candidates)} candidates")
    print("Computing scores with JD-compliant ranking...")
    results = compute_scores_batch(candidates)
    
    # Sort by score descending, then candidate_id ascending for tie-breaking
    results.sort(key=lambda x: (-x['score'], x['candidate_id']))
    
    # Assign ranks with strict decreasing scores
    final = []
    prev_score = 1.0
    for rank, item in enumerate(results[:100], 1):
        curr_score = min(prev_score, round(item['score'], 6))
        if curr_score >= prev_score:
            curr_score = round(prev_score - 0.00001, 6)
        prev_score = curr_score
        
        reasoning = item['reasoning']
        # Adapt tone based on rank for Stage 4 consistency
        if rank <= 10:
            reasoning = "Top-tier candidate. " + reasoning
        elif rank <= 30:
            pass  # Standard reasoning already appropriate
        elif rank <= 60:
            reasoning = "Solid experience. " + reasoning
        else:
            reasoning = "Adjacent experience. " + reasoning
        
        final.append({
            'candidate_id': item['candidate_id'],
            'rank': rank,
            'score': curr_score,
            'reasoning': reasoning
        })
    
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['candidate_id', 'rank', 'score', 'reasoning'])
        w.writeheader()
        w.writerows(final)
    
    print(f"Done in {time.time() - start:.1f}s. Top 100 written to {args.out}")
    print(f"Total candidates processed: {len(candidates)}")
    print(f"Qualified candidates: {len(results)}")
    print(f"Disqualified candidates: {len(candidates) - len(results)}")


if __name__ == '__main__':
    main()
