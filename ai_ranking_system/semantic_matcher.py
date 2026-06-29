"""
Semantic Similarity using TF-IDF (fallback without model download).
For CPU-only environments with no network.
"""

import numpy as np
from typing import Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticMatcher:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.job_vector = None
        self.fitted = False

    def fit_and_transform_job(self, job_text: str) -> None:
        self.vectorizer.fit([job_text])
        self.job_vector = self.vectorizer.transform([job_text])
        self.fitted = True

    def compute_candidate_similarity(self, candidate: Dict) -> float:
        if not self.fitted:
            return 0.0
        
        profile = candidate.get('profile', {})
        career_history = candidate.get('career_history', [])
        
        text_parts = []
        text_parts.append(profile.get('headline', ''))
        text_parts.append(profile.get('summary', ''))
        
        for role in career_history[:3]:
            text_parts.append(role.get('title', ''))
            desc = role.get('description', '')
            if desc:
                text_parts.append(desc[:500])
        
        for skill in candidate.get('skills', [])[:15]:
            text_parts.append(skill.get('name', ''))
        
        combined_text = ' '.join(filter(None, text_parts))
        
        if not combined_text.strip():
            return 0.0
        
        candidate_vector = self.vectorizer.transform([combined_text])
        similarity = cosine_similarity(candidate_vector, self.job_vector)[0][0]
        
        return float(similarity)