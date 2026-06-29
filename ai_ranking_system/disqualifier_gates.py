"""
JD-Specific Disqualifier Gates for Senior AI Engineer Role.
Implements hard exclusions and pre-score filters aligned with the Redrob JD.
"""

import re
from typing import Dict, List, Tuple

CONSULTING_FIRMS = {
    'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini',
    'mindtree', 'tech mahindra', 'hcl', 'lti', 'persistent'
}

NON_TECHNICAL_TITLES = {
    'marketing manager', 'sales', 'operations manager', 'project manager',
    'hr manager', 'accountant', 'content writer', 'graphic designer',
    'civil engineer', 'mechanical engineer', 'customer support',
    'business analyst', 'sales executive', 'operations'
}

DISQUALIFYING_PRIMARY_EXPERTISE = {
    'computer vision', 'cv engineer', 'vision engineer', 'robotics',
    'speech engineer', 'asr engineer', 'tts engineer'
}

SIDE_PROJECT_INDICATORS = [
    r'\b(?:langchain|openai)\s+(?:api|tutorial|side project|experiment)\b',
    r'\b(?:side project|online course|taking courses)\b.*\b(?:ai|ml|rag|llm)\b',
    r'\b(?:curious about|exploring|getting into)\b.*\b(?:ai|ml|genai)\b',
    r'\b(?:recently|last few months?)\b.*\b(?:langchain|openai|rag)\b'
]

PRODUCTION_INDICATORS = [
    r'\b(?:deployed|production|shipped|live|real.?users|real.?world)\b',
    r'\b(?:latency|p95|p99|throughput|scale|serving)\b',
    r'\b(?:a/b test|experiment|evaluation|metric|ndcg|mrr|map)\b',
    r'\b(?:hybrid|dense|sparse|vector|embedding)\b.*\b(?:retrieval|search|ranking)\b'
]


class DisqualifierGates:
    """
    Hard disqualifier gates that filter candidates BEFORE scoring.
    These represent the explicit 'do NOT want' criteria from the JD.
    """

    def __init__(self):
        self.side_project_patterns = [re.compile(p, re.IGNORECASE) for p in SIDE_PROJECT_INDICATORS]
        self.production_patterns = [re.compile(p, re.IGNORECASE) for p in PRODUCTION_INDICATORS]

    def check_all_gates(self, candidate: Dict) -> Tuple[bool, str, float]:
        """
        Returns (is_disqualified, reason, penalty_multiplier).
        If is_disqualified is True, the candidate should be excluded from ranking.
        If False but penalty_multiplier < 1.0, the candidate is heavily penalized.
        """
        profile = candidate.get('profile', {})
        career = candidate.get('career_history', [])
        skills = candidate.get('skills', [])
        signals = candidate.get('redrob_signals', {})
        
        title = profile.get('current_title', '').lower()
        summary = profile.get('summary', '').lower()
        headline = profile.get('headline', '').lower()
        
        skill_names = [s.get('name', '').lower() for s in skills]
        
        combined_text = f"{headline} {summary} " + ' '.join(
            r.get('description', '') + ' ' + r.get('title', '') for r in career
        ).lower()
        
        # Gate 1: Non-technical title with AI keyword stuffing and no production IR/Search/Ranking experience
        if self._is_non_technical_title(title):
            has_ir_production = self._has_ir_production_experience(combined_text)
            if not has_ir_production:
                return True, f"Non-technical title '{profile.get('current_title', '')}' with no production search/ranking/retrieval experience", 0.0
        
        # Gate 2: Civil/Mechanical/other non-software engineering primary
        if self._is_non_software_engineering(title, skill_names, combined_text):
            return True, f"Primary expertise is non-software engineering (Civil/Mechanical/etc.) without NLP/IR/Search/Ranking production experience", 0.0
        
        # Gate 3: Computer Vision / Speech / Robotics primary without NLP/IR exposure
        if self._is_cv_speech_robotics_primary(title, skill_names, combined_text, career):
            return True, f"Primary expertise is Computer Vision/Speech/Robotics without significant NLP/IR/Search/Ranking production experience", 0.0
        
        # Gate 4: Consulting-only background with no product company experience
        if self._is_consulting_only(career):
            return True, "Consulting-only background (TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini/etc.) with no product company experience", 0.0
        
        # Gate 5: LangChain/OpenAI side projects as PRIMARY experience (not supplemented by production)
        if self._is_langchain_side_project_primary(summary, combined_text, career):
            return True, "AI experience consists primarily of recent LangChain/OpenAI side projects without substantial pre-LLM production ML experience", 0.0
        
        # Gate 6: Pure research background without production deployment
        if self._is_pure_research_background(career, combined_text):
            return True, "Career in pure research environments without production deployment", 0.0
        
        return False, "", 1.0

    def _is_non_technical_title(self, title: str) -> bool:
        return any(t in title for t in NON_TECHNICAL_TITLES)

    def _is_non_software_engineering(self, title: str, skill_names: List[str], text: str) -> bool:
        non_software = ['civil engineer', 'mechanical engineer', 'electrical engineer', 'chemical engineer']
        if any(t in title for t in non_software):
            has_software_exp = any(t in text for t in [
                'python', 'ml', 'machine learning', 'nlp', 'software', 'deployed',
                'production', 'api', 'algorithm', 'model'
            ])
            if not has_software_exp:
                return True
        return False

    def _is_cv_speech_robotics_primary(self, title: str, skill_names: List[str], text: str, career: List[Dict]) -> bool:
        cv_terms = ['computer vision', 'cv engineer', 'vision engineer', 'robotics',
                   'speech engineer', 'asr engineer', 'tts engineer']
        is_cv_primary = any(t in title for t in cv_terms)
        
        if not is_cv_primary:
            return False
        
        # Check if MOST RECENT role is ALSO in CV/Speech/Robotics
        recent_cv = False
        recent_desc = ''
        if career:
            recent_title = career[0].get('title', '').lower()
            recent_cv = any(t in recent_title for t in cv_terms)
            recent_desc = career[0].get('description', '').lower()
        
        # If current role is NOT CV, they may have pivoted to search/ranking
        if not recent_cv:
            return False
        
        # Count CV-related skills
        cv_skill_count = sum(1 for s in skill_names if any(t in s for t in [
            'computer vision', 'object detection', 'image classification', 'yolo',
            'opencv', 'cnn', 'gan', 'diffusion', 'segmentation', 'detection'
        ]))
        
        # Look for EXPLICIT production search/ranking/retrieval system in MOST RECENT role
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
        ]
        has_recent_search = any(re.search(p, recent_desc, re.IGNORECASE) for p in recent_role_patterns)
        
        # If CV primary with >= 2 CV skills and no explicit production search/ranking/retrieval system → disqualify
        if cv_skill_count >= 2 and not has_recent_search:
            return True
        
        # If current role is CV AND most recent role has no search/ranking/retrieval system → disqualify
        if recent_cv and not has_recent_search:
            return True
        
        return False

    def _is_consulting_only(self, career: List[Dict]) -> bool:
        """Check if ALL career roles are at consulting firms."""
        if not career:
            return False
        
        for role in career:
            company = role.get('company', '').lower()
            duration = role.get('duration_months', 0)
            if duration < 12:
                continue
            if company not in CONSULTING_FIRMS:
                return False
        return True

    def _is_langchain_side_project_primary(self, summary: str, text: str, career: List[Dict]) -> bool:
        """Detect if AI experience is primarily LangChain/OpenAI side projects."""
        side_project_signals = sum(1 for p in self.side_project_patterns if p.search(summary))
        production_signals = sum(1 for p in self.production_patterns if p.search(text))
        
        has_real_jobs = len([r for r in career if r.get('duration_months', 0) >= 12]) >= 1
        
        if side_project_signals >= 2 and production_signals <= 1 and has_real_jobs:
            return True
        
        if 'civil engineer' in summary and ('langchain' in summary or 'rag' in summary or 'openai' in summary):
            return True
        
        return False

    def _is_pure_research_background(self, career: List[Dict], text: str) -> bool:
        research_keywords = ['phd', 'research scientist', 'research engineer', 'postdoc', 'academic']
        production_keywords = ['deployed', 'production', 'shipped', 'live', 'real users']
        
        is_research = any(
            any(k in r.get('title', '').lower() for k in research_keywords) or
            any(k in r.get('company', '').lower() for k in ['university', 'institute', 'lab'])
            for r in career[:2]
        )
        
        has_production = any(k in text for k in production_keywords)
        
        if is_research and not has_production:
            return True
        
        return False

    def _has_ir_production_experience(self, text: str) -> bool:
        """Check for production IR/Search/Ranking experience."""
        ir_prod_terms = [
            'retrieval', 'search', 'ranking', 'recommendation', 'vector search',
            'semantic search', 'hybrid search', 'embedding', 'bm25', 'elasticsearch',
            'opensearch', 'pinecone', 'milvus', 'faiss', 'weaviate', 'qdrant',
            'ndcg', 'mrr', 'map', 'learning-to-rank', 'rerank'
        ]
        return any(term in text for term in ir_prod_terms)

    def apply_penalty_multipliers(self, candidate: Dict, base_score: float) -> Tuple[float, str]:
        """Apply penalty multipliers for soft disqualifiers."""
        profile = candidate.get('profile', {})
        career = candidate.get('career_history', [])
        signals = candidate.get('redrob_signals', {})
        
        penalty = 1.0
        reasons = []
        
        # Penalty for non-ideal titles without IR production experience
        title = profile.get('current_title', '').lower()
        if self._is_non_technical_title(title):
            penalty *= 0.7
            reasons.append("Non-technical title penalty (-30%)")
        
        # Penalty for high notice period (>90 days) - JD prefers sub-30 days
        notice = signals.get('notice_period_days', 90)
        if notice > 90:
            penalty *= 0.85
            reasons.append(f"High notice period {notice}d (-15%)")
        
        # Penalty for low recruiter response rate
        resp_rate = signals.get('recruiter_response_rate', 0.5)
        if resp_rate < 0.3:
            penalty *= 0.6
            reasons.append(f"Low response rate {resp_rate:.0%} (-40%)")
        elif resp_rate < 0.5:
            penalty *= 0.8
            reasons.append(f"Below-average response rate {resp_rate:.0%} (-20%)")
        
        # Penalty for stale profile
        last_active = signals.get('last_active_date', '')
        if last_active:
            from datetime import datetime
            try:
                ref = datetime.strptime(last_active, "%Y-%m-%d")
                days_inactive = (datetime(2026, 6, 26) - ref).days
                if days_inactive > 180:
                    penalty *= 0.5
                    reasons.append(f"Stale profile ({days_inactive}d inactive) (-50%)")
                elif days_inactive > 90:
                    penalty *= 0.8
                    reasons.append(f"Inactive {days_inactive}d (-20%)")
            except:
                pass
        
        # Penalty for not open to work
        if not signals.get('open_to_work_flag', False):
            penalty *= 0.7
            reasons.append("Not open to work (-30%)")
        
        # Penalty for job hopping (switching every ~1.5 years)
        if len(career) >= 3:
            avg_tenure = sum(r.get('duration_months', 0) for r in career) / len(career)
            if avg_tenure < 18:
                penalty *= 0.85
                reasons.append(f"Frequent job changes (avg {avg_tenure:.0f} months) (-15%)")
        
        return base_score * penalty, "; ".join(reasons)