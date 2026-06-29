"""
Honeypot Detector - Identifies suspicious profiles with impossible timelines or skill claims.
Implements JD-specific disqualifiers for Senior AI Engineer role.
"""

import re
from datetime import datetime
from typing import Dict, Tuple, List

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
]


class HoneypotDetector:
    def __init__(self):
        self.suspicious_patterns = [
            r'\b(langchain|llm|rag|ai)\s+tutorial',
            r'hot framework',
            r'prompt.?engineering',
            r'building with ai',
            r'ai enthusiast',
            r'curious about ai',
        ]
        self.side_project_patterns = [re.compile(p, re.IGNORECASE) for p in SIDE_PROJECT_INDICATORS]

    def detect_timeline_anomalies(self, candidate: Dict) -> float:
        penalty = 0.0
        career_history = candidate.get('career_history', [])
        
        for role in career_history:
            duration = role.get('duration_months', 0)
            if duration < 0 or duration > 360:
                penalty += 0.3
            
            if duration > 600:
                penalty += 0.5
        
        return min(penalty, 1.0)

    def detect_skill_inconsistencies(self, candidate: Dict) -> float:
        penalty = 0.0
        skills = candidate.get('skills', [])
        
        expert_skills_count = sum(1 for s in skills if s.get('proficiency') == 'expert')
        advanced_skills_count = sum(1 for s in skills if s.get('proficiency') == 'advanced')
        
        if expert_skills_count > 8:
            penalty += 0.4
        if advanced_skills_count > 12 and len(skills) > 15:
            penalty += 0.3
        
        for skill in skills:
            duration_months = skill.get('duration_months', 0)
            proficiency = skill.get('proficiency', '')
            if proficiency in ['expert', 'advanced'] and duration_months < 6:
                penalty += 0.1
        
        return min(penalty, 1.0)

    def detect_title_mismatch(self, candidate: Dict) -> float:
        penalty = 0.0
        profile = candidate.get('profile', {})
        skills = candidate.get('skills', [])
        
        current_title = profile.get('current_title', '').lower()
        summary = profile.get('summary', '').lower()
        
        has_ai_skills = any(
            term in skill.get('name', '').lower() 
            for skill in skills 
            for term in ['ml', 'nlp', 'embedding', 'retrieval', 'llm', 'rag', 'vector', 'pytorch', 'tensorflow']
        )
        
        title_indicates_non_ai = any(
            prohibited in current_title.lower() 
            for prohibited in ['marketing', 'sales', 'operations', 'project', 'hr', 'accountant', 'content writer', 'graphic designer']
        )
        
        if has_ai_skills and title_indicates_non_ai:
            if 'transition' not in summary and 'curious' not in summary and 'side project' not in summary:
                penalty += 0.2
        
        return min(penalty, 0.5)

    def detect_background_mismatch(self, candidate: Dict) -> float:
        penalty = 0.0
        career_history = candidate.get('career_history', [])
        summary = candidate.get('profile', {}).get('summary', '').lower()
        
        consulting_only = True
        for role in career_history:
            company = role.get('company', '').lower()
            duration = role.get('duration_months', 0)
            if duration < 12:
                continue
            if company not in CONSULTING_FIRMS:
                consulting_only = False
                break
        
        if consulting_only:
            penalty += 0.5
        
        if 'tutorial' in summary or 'how i used' in summary:
            penalty += 0.3
        
        return min(penalty, 1.0)

    def detect_cv_speech_robotics_primary(self, candidate: Dict) -> float:
        """Detect if primary expertise is CV/Speech/Robotics without NLP/IR exposure."""
        penalty = 0.0
        profile = candidate.get('profile', {})
        career = candidate.get('career_history', [])
        skills = candidate.get('skills', [])
        
        title = profile.get('current_title', '').lower()
        summary = profile.get('summary', '').lower()
        skill_names = [s.get('name', '').lower() for s in skills]
        
        cv_terms = ['computer vision', 'cv engineer', 'vision engineer', 'robotics',
                   'speech engineer', 'asr engineer', 'tts engineer']
        is_cv_primary = any(t in title for t in cv_terms)
        
        if not is_cv_primary:
            return 0.0
        
        cv_skill_count = sum(1 for s in skill_names if any(t in s for t in [
            'computer vision', 'object detection', 'image classification', 'yolo',
            'opencv', 'cnn', 'gan', 'diffusion', 'segmentation', 'detection'
        ]))
        
        text = ' '.join([
            summary,
            ' '.join(r.get('description', '') + ' ' + r.get('title', '') for r in career)
        ]).lower()
        
        nlp_ir_terms = ['nlp', 'retrieval', 'search', 'ranking', 'recommendation',
                       'embedding', 'vector', 'semantic', 'rag', 'information retrieval',
                       'ndcg', 'mrr', 'map', 'learning-to-rank']
        has_nlp_ir = any(term in text for term in nlp_ir_terms)
        
        if cv_skill_count >= 3 and not has_nlp_ir:
            penalty += 0.8
        
        return min(penalty, 1.0)

    def detect_langchain_side_project_primary(self, candidate: Dict) -> float:
        """Detect if AI experience is primarily recent LangChain/OpenAI side projects."""
        penalty = 0.0
        profile = candidate.get('profile', {})
        career = candidate.get('career_history', [])
        summary = profile.get('summary', '').lower()
        title = profile.get('current_title', '').lower()
        
        side_signals = sum(1 for p in self.side_project_patterns if p.search(summary))
        
        text = ' '.join([
            summary,
            ' '.join(r.get('description', '') + ' ' + r.get('title', '') for r in career)
        ]).lower()
        
        production_signals = sum(1 for kw in ['deployed', 'production', 'shipped', 'live', 
                                              'real users', 'latency', 'throughput', 'a/b test']
                                if kw in text)
        
        has_real_jobs = len([r for r in career if r.get('duration_months', 0) >= 12]) >= 1
        
        if side_signals >= 2 and production_signals <= 1 and has_real_jobs:
            penalty += 0.7
        
        if 'civil engineer' in title and ('langchain' in summary or 'rag' in summary or 'openai' in summary):
            penalty += 0.9
        
        return min(penalty, 1.0)

    def detect_non_software_engineering_primary(self, candidate: Dict) -> float:
        """Detect non-software engineering primary (Civil, Mechanical, etc.)."""
        penalty = 0.0
        profile = candidate.get('profile', {})
        title = profile.get('current_title', '').lower()
        
        non_software = ['civil engineer', 'mechanical engineer', 'electrical engineer', 'chemical engineer']
        is_non_software = any(t in title for t in non_software)
        
        if not is_non_software:
            return 0.0
        
        text = ' '.join([
            profile.get('summary', ''),
            ' '.join(r.get('description', '') for r in candidate.get('career_history', []))
        ]).lower()
        
        has_software_exp = any(kw in text for kw in [
            'python', 'ml', 'machine learning', 'nlp', 'software', 'deployed',
            'production', 'api', 'algorithm', 'model', 'engineer'
        ])
        
        if not has_software_exp:
            penalty += 0.9
        elif 'side project' in text or 'course' in text or 'exploring' in text:
            penalty += 0.7
        
        return min(penalty, 1.0)

    def is_honeypot(self, candidate: Dict) -> Tuple[bool, float, str]:
        total_penalty = 0.0
        reasons = []
        
        timeline_penalty = self.detect_timeline_anomalies(candidate)
        if timeline_penalty > 0.1:
            reasons.append(f"timeline anomaly (+{timeline_penalty:.1f})")
            total_penalty += timeline_penalty
        
        skill_penalty = self.detect_skill_inconsistencies(candidate)
        if skill_penalty > 0.1:
            reasons.append(f"skill inconsistency (+{skill_penalty:.1f})")
            total_penalty += skill_penalty
        
        title_penalty = self.detect_title_mismatch(candidate)
        if title_penalty > 0.1:
            reasons.append(f"title/skill mismatch (+{title_penalty:.1f})")
            total_penalty += title_penalty
        
        background_penalty = self.detect_background_mismatch(candidate)
        if background_penalty > 0.1:
            reasons.append(f"consulting-only background (+{background_penalty:.1f})")
            total_penalty += background_penalty
        
        cv_penalty = self.detect_cv_speech_robotics_primary(candidate)
        if cv_penalty > 0.1:
            reasons.append(f"CV/Speech/Robotics primary without NLP/IR (+{cv_penalty:.1f})")
            total_penalty += cv_penalty
        
        langchain_penalty = self.detect_langchain_side_project_primary(candidate)
        if langchain_penalty > 0.1:
            reasons.append(f"LangChain/OpenAI side projects primary (+{langchain_penalty:.1f})")
            total_penalty += langchain_penalty
        
        non_software_penalty = self.detect_non_software_engineering_primary(candidate)
        if non_software_penalty > 0.1:
            reasons.append(f"Non-software engineering primary (+{non_software_penalty:.1f})")
            total_penalty += non_software_penalty
        
        is_suspicious = total_penalty > 0.5
        reason_str = "; ".join(reasons) if reasons else ""
        
        return is_suspicious, total_penalty, reason_str
