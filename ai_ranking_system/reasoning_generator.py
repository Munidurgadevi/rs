"""
Candidate Reasoning Generator.
Creates human-readable explanations for rankings with specific, evidence-based reasoning.
"""

from typing import Dict, Optional, List

class ReasoningGenerator:
    def __init__(self):
        self.search_terms = ['retrieval', 'search', 'ranking', 'embedding', 'vector', 'milvus', 
                           'pinecone', 'faiss', 'elasticsearch', 'rag', 'sentence-transformers',
                           'recommendation', 'hybrid search', 'semantic search']
        self.ml_terms = ['production', 'deployed', 'ml', 'machine learning', 'nlp', 'pytorch', 
                        'tensorflow', 'scikit-learn', 'xgboost', 'fine-tuning', 'bge', 'e5']

    def generate_reasoning(self, candidate: Dict, score: float) -> str:
        profile = candidate.get('profile', {})
        signals = candidate.get('redrob_signals', {})
        skills = candidate.get('skills', [])
        career_history = candidate.get('career_history', [])
        
        years_exp = profile.get('years_of_experience', 0)
        current_title = profile.get('current_title', '')
        location = profile.get('location', '')
        
        ai_skills = [s.get('name') for s in skills if self._is_ai_skill(s.get('name'))]
        core_skills = [s.get('name') for s in skills if s.get('proficiency') in ['advanced', 'expert']]
        
        response_rate = signals.get('recruiter_response_rate', 0)
        saved_by_recruiters = signals.get('saved_by_recruiters_30d', 0)
        open_to_work = signals.get('open_to_work_flag', False)
        notice_period = signals.get('notice_period_days', 90)
        last_active = signals.get('last_active_date', '')
        
        reasons = []
        
        # Experience
        if 5 <= years_exp <= 9:
            reasons.append(f"{int(years_exp)}yrs")
        elif years_exp > 9:
            reasons.append(f"{int(years_exp)}yrs (senior)")
        elif years_exp >= 4:
            reasons.append(f"{int(years_exp)}yrs (near target)")
        else:
            reasons.append(f"{int(years_exp)}yrs (junior)")
        
        # Skills depth
        if len(ai_skills) >= 5:
            reasons.append(f"{len(ai_skills)} core AI skills")
        elif len(ai_skills) >= 3:
            reasons.append(f"{len(ai_skills)} AI skills")
        elif len(ai_skills) >= 1:
            reasons.append(f"{len(ai_skills)} relevant skills")
        
        # Production systems built
        systems_built = []
        for role in career_history[:3]:
            desc = role.get('description', '').lower()
            company = role.get('company', '')
            if any(term in desc for term in self.search_terms):
                if any(prod in desc for prod in ['production', 'deployed', 'live', 'real-users', 'serving']):
                    systems_built.append(f"built search/ranking at {company}")
                    break
        if systems_built:
            reasons.append(systems_built[0])
        
        # Evaluation framework
        eval_text = ' '.join(r.get('description', '').lower() for r in career_history)
        if any(term in eval_text for term in ['ndcg', 'mrr', 'map', 'a/b test', 'evaluation framework']):
            reasons.append("eval framework experience")
        
        # Behavioral signals
        if response_rate >= 0.7:
            reasons.append(f"response {response_rate:.0%}")
        elif response_rate >= 0.5:
            reasons.append(f"response {response_rate:.0%}")
        
        if saved_by_recruiters >= 10:
            reasons.append(f"saved {saved_by_recruiters}x")
        
        # Availability
        if open_to_work:
            if notice_period <= 30:
                reasons.append(f"available ({notice_period}d notice)")
            elif notice_period <= 60:
                reasons.append(f"available ({notice_period}d notice)")
            else:
                reasons.append("open to work")
        
        # Location
        if any(loc in location.lower() for loc in ['pune', 'noida', 'hyderabad', 'mumbai', 'delhi', 'bangalore']):
            reasons.append("India-based")
        
        # Sentiment
        if score >= 0.85:
            sentiment = "strong fit"
        elif score >= 0.7:
            sentiment = "good fit"
        elif score >= 0.55:
            sentiment = "potential fit"
        else:
            sentiment = "weak fit"
        
        # Build reasoning string
        if reasons:
            reasoning = f"{current_title}: {', '.join(reasons[:5])}; {sentiment}"
        else:
            reasoning = f"{current_title}; {sentiment}"
        
        return reasoning[:200]

    def _is_ai_skill(self, skill_name: str) -> bool:
        skill_lower = skill_name.lower()
        return any(term in skill_lower for term in self.search_terms + self.ml_terms)
    
    def generate_detailed_reasoning(self, candidate: Dict, score: float, 
                                    disqualifiers: List[str] = None) -> str:
        """Generate detailed reasoning including disqualifier warnings."""
        base_reasoning = self.generate_reasoning(candidate, score)
        
        if disqualifiers:
            warnings = "; ".join(disqualifiers[:2])
            return f"{base_reasoning} [FLAG: {warnings}]"
        
        return base_reasoning