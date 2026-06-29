"""
Embedding Similarity Scoring Module.
Uses sentence-transformers for semantic similarity between job description and candidate profiles.
"""

import numpy as np
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingRanker:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.job_embedding = None
        self.candidate_embeddings: Dict[str, np.ndarray] = {}

    def compute_job_embedding(self, job_text: str) -> np.ndarray:
        self.job_embedding = self.model.encode(job_text, normalize_embeddings=True)
        return self.job_embedding

    def compute_candidate_embedding(self, candidate: Dict) -> np.ndarray:
        profile = candidate.get('profile', {})
        career_history = candidate.get('career_history', [])
        
        text_parts = []
        
        text_parts.append(profile.get('headline', ''))
        text_parts.append(profile.get('summary', ''))
        
        for role in career_history[:4]:
            text_parts.append(role.get('title', ''))
            desc = role.get('description', '')
            if desc:
                text_parts.append(desc[:500])
        
        combined_text = ' '.join(filter(None, text_parts))
        
        return self.model.encode(combined_text, normalize_embeddings=True)

    def compute_semantic_similarity(self, candidate_embedding: np.ndarray) -> float:
        if self.job_embedding is None:
            return 0.0
        
        sim = cosine_similarity(
            candidate_embedding.reshape(1, -1),
            self.job_embedding.reshape(1, -1)
        )[0][0]
        
        return float(sim)

    def get_embedding_text(self, candidate: Dict) -> str:
        profile = candidate.get('profile', {})
        career_history = candidate.get('career_history', [])
        
        text_parts = []
        text_parts.append(profile.get('headline', ''))
        text_parts.append(profile.get('summary', ''))
        
        for role in career_history[:3]:
            text_parts.append(role.get('title', ''))
            desc = role.get('description', '')
            if desc:
                text_parts.append(desc[:300])
        
        for skill in candidate.get('skills', [])[:10]:
            text_parts.append(skill.get('name', ''))
        
        return ' '.join(filter(None, text_parts))