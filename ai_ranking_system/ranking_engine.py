"""
Hybrid Ranking Engine.
Combines semantic similarity, features, and behavioral signals.
Fully compliant with Senior AI Engineer JD requirements.
"""

import json
from typing import Dict, List, Tuple
from tqdm import tqdm

from job_description_analyzer import JobDescriptionAnalyzer
from honeypot_detector import HoneypotDetector
from feature_extractor import FeatureExtractor
from behavioral_scorer import BehavioralSignalScorer
from disqualifier_gates import DisqualifierGates
from evaluation_framework import EvaluationFramework
from vector_db_simulator import VectorDBSimulator

class RankingEngine:
    def __init__(self):
        self.job_analyzer = JobDescriptionAnalyzer()
        self.honeypot_detector = HoneypotDetector()
        self.feature_extractor = FeatureExtractor()
        self.behavioral_scorer = BehavioralSignalScorer()
        self.disqualifier_gates = DisqualifierGates()
        self.eval_framework = EvaluationFramework()
        self.vector_db = VectorDBSimulator()
        
        self.weights = {
            'semantic_similarity': 0.20,
            'skill_match': 0.15,
            'experience_relevance': 0.15,
            'career_trajectory': 0.10,
            'behavioral_signals': 0.10,
            'availability': 0.05,
            'evaluation_framework': 0.15,
            'company_quality': 0.10
        }

    def compute_skill_match_score(self, candidate: Dict) -> float:
        skills = candidate.get('skills', [])
        skill_names = [s.get('name', '').lower() for s in skills]
        
        core_ai_skills = 0
        python_skills = 0
        total_relevant = 0
        
        for skill_name in skill_names:
            if any(term in skill_name for term in ['embedding', 'retrieval', 'ranking', 'vector', 
                                                   'search', 'milvus', 'pinecone', 'faiss', 
                                                   'nlp', 'llm', 'rag', 'sentence-transformers',
                                                   'bge', 'e5', 'learning-to-rank', 'recommender', 'recommendation']):
                core_ai_skills += 1
                total_relevant += 1
            elif skill_name in ['python', 'pytorch', 'tensorflow', 'scikit-learn', 'numpy', 'pandas']:
                python_skills += 1
                total_relevant += 1
            elif any(term in skill_name for term in ['machine learning', 'ml', 'deep learning', 'xgboost']):
                total_relevant += 1
        
        score = 0.0
        if core_ai_skills >= 3:
            score += 0.5
        elif core_ai_skills >= 1:
            score += 0.25
        
        if python_skills >= 1:
            score += 0.3
        
        if total_relevant >= 5:
            score += 0.2
        
        return min(1.0, score)

    def rank_candidate(self, candidate: Dict, semantic_sim: float) -> Tuple[float, str, bool]:
        """
        Rank a single candidate with JD-compliant scoring.
        Returns (score, reasoning, is_disqualified)
        """
        # Step 1: Check disqualifier gates
        is_disqualified, disqualify_reason, _ = self.disqualifier_gates.check_all_gates(candidate)
        if is_disqualified:
            return 0.0, f"DISQUALIFIED: {disqualify_reason}", True
        
        # Step 2: Honeypot detection
        honeypot, honeypot_penalty, honeypot_reasons = self.honeypot_detector.is_honeypot(candidate)
        
        # Step 3: Compute component scores
        skill_score = self.compute_skill_match_score(candidate)
        exp_relevance = self.feature_extractor.extract_production_ml_score(candidate)
        exp_score = self.feature_extractor.extract_experience_score(candidate)
        career_traj = self.feature_extractor.extract_career_trajectory_score(candidate)
        behavior = self.behavioral_scorer.compute_total_behavioral_score(candidate)
        availability = self.behavioral_scorer.compute_availability_score(candidate.get('redrob_signals', {}))
        eval_score = self.eval_framework.score_evaluation_framework_experience(candidate)
        company_quality = self.feature_extractor.extract_company_quality_score(candidate)
        
        # Step 4: Weighted combination with JD-aligned weights
        base_score = (
            self.weights['semantic_similarity'] * max(0, semantic_sim) +
            self.weights['skill_match'] * skill_score +
            self.weights['experience_relevance'] * exp_relevance +
            self.weights['career_trajectory'] * career_traj +
            self.weights['behavioral_signals'] * behavior +
            self.weights['availability'] * availability +
            self.weights['evaluation_framework'] * eval_score +
            self.weights['company_quality'] * company_quality
        )
        
        # Step 5: Apply experience multiplier (bell curve 5-9 years)
        base_score *= exp_score
        
        # Step 6: Apply product company penalty
        product_score = self.feature_extractor.extract_product_company_score(candidate)
        if product_score < 0.3:
            base_score *= 0.7
        
        # Step 7: Apply honeypot penalty
        if honeypot:
            base_score *= (1 - min(0.8, honeypot_penalty))
        
        # Step 8: Apply disqualifier gate penalties
        final_score, penalty_reasons = self.disqualifier_gates.apply_penalty_multipliers(candidate, base_score)
        
        # Step 9: Clamp to [0, 1]
        final_score = max(0.0, min(1.0, final_score))
        
        # Step 10: Generate reasoning
        all_reasons = []
        if honeypot_reasons:
            all_reasons.append(honeypot_reasons)
        if penalty_reasons:
            all_reasons.append(penalty_reasons)
        
        from reasoning_generator import ReasoningGenerator
        reasoning_gen = ReasoningGenerator()
        reasoning = reasoning_gen.generate_detailed_reasoning(
            candidate, final_score, all_reasons if all_reasons else None
        )
        
        return final_score, reasoning, False

    def process_candidates(self, candidates_path: str, embedding_ranker=None) -> List[Tuple[str, int, float, str]]:
        results = []
        
        with open(candidates_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing candidates"):
                candidate = json.loads(line.strip())
                candidate_id = candidate.get('candidate_id', '')
                
                # Use vector DB simulator for hybrid search if available
                if embedding_ranker is not None:
                    embedding = embedding_ranker.compute_candidate_embedding(candidate)
                    semantic_sim = embedding_ranker.compute_semantic_similarity(embedding)
                else:
                    # Fallback to simple semantic similarity
                    from .semantic_matcher import SemanticMatcher
                    matcher = SemanticMatcher()
                    matcher.fit_and_transform_job("Senior AI Engineer production ML systems embeddings retrieval ranking vector databases search infrastructure Python product company")
                    semantic_sim = matcher.compute_candidate_similarity(candidate)
                
                score, reasoning, is_disqualified = self.rank_candidate(candidate, semantic_sim)
                
                if not is_disqualified:
                    results.append((candidate_id, score, reasoning))
        
        results.sort(key=lambda x: (-x[1], x[0]))
        
        ranked_results = []
        for rank, (candidate_id, score, reasoning) in enumerate(results[:100], start=1):
            ranked_results.append((candidate_id, rank, score, reasoning))
        
        return ranked_results
