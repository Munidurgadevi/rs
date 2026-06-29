"""
Main Pipeline Orchestrator for AI Candidate Discovery & Ranking System.
"""

import json
import csv
import time
import os
from typing import Dict, List
from tqdm import tqdm
import numpy as np

from .honeypot_detector import HoneypotDetector
from .feature_extractor import FeatureExtractor
from .behavioral_scorer import BehavioralSignalScorer
from .semantic_matcher import SemanticMatcher
from .reasoning_generator import ReasoningGenerator

class CandidateRankingPipeline:
    def __init__(self):
        self.honeypot_detector = HoneypotDetector()
        self.feature_extractor = FeatureExtractor()
        self.behavioral_scorer = BehavioralSignalScorer()
        self.reasoning_generator = ReasoningGenerator()
        self.semantic_matcher = SemanticMatcher()
        
        self.weights = {
            'semantic_similarity': 0.40,
            'skill_match': 0.20,
            'experience_relevance': 0.15,
            'career_trajectory': 0.10,
            'behavioral_signals': 0.10,
            'availability': 0.05
        }

    def compute_skill_match_score(self, candidate: Dict) -> float:
        skills = candidate.get('skills', [])
        skill_names = [s.get('name', '').lower() for s in skills]
        core_ai_skills = sum(1 for s in skill_names 
                          if any(t in s for t in ['embedding', 'retrieval', 'ranking', 'vector', 
                                                     'search', 'milvus', 'pinecone', 'faiss', 
                                                     'nlp', 'llm', 'rag', 'bge', 'e5',
                                                     'learning-to-rank', 'recommender', 'recommendation']))
        python_skills = sum(1 for s in skill_names if s == 'python' or any(t in s for t in ['pytorch', 'tensorflow', 'scikit']))
        
        score = 0.0
        if core_ai_skills >= 4:
            score += 0.5
        elif core_ai_skills >= 2:
            score += 0.35
        elif core_ai_skills >= 1:
            score += 0.2
        if python_skills >= 1:
            score += 0.3
        return min(1.0, score)

    def rank_candidate(self, candidate: Dict, semantic_sim: float) -> float:
        honeypot, honeypot_penalty, _ = self.honeypot_detector.is_honeypot(candidate)
        
        skill_score = self.compute_skill_match_score(candidate)
        exp_relevance = self.feature_extractor.extract_production_ml_score(candidate)
        career_traj = self.feature_extractor.extract_career_trajectory_score(candidate)
        behavior = self.behavioral_scorer.compute_total_behavioral_score(candidate)
        availability = self.behavioral_scorer.compute_availability_score(candidate.get('redrob_signals', {}))
        
        base_score = (
            self.weights['semantic_similarity'] * max(0, semantic_sim) +
            self.weights['skill_match'] * skill_score +
            self.weights['experience_relevance'] * exp_relevance +
            self.weights['career_trajectory'] * career_traj +
            self.weights['behavioral_signals'] * behavior +
            self.weights['availability'] * availability
        )
        
        product_score = self.feature_extractor.extract_product_company_score(candidate)
        if product_score < 0.2:
            base_score *= 0.5
        
        if honeypot:
            base_score *= (1 - min(0.8, honeypot_penalty))
        
        return max(0.0, min(1.0, base_score))

    def run(self, candidates_path: str, output_path: str) -> List[Dict]:
        start_time = time.time()
        
        job_text = "Senior AI Engineer production ML systems embeddings retrieval ranking vector databases search infrastructure Python product company"
        self.semantic_matcher.fit_and_transform_job(job_text)
        
        print(f"Processing candidates from {candidates_path}...")
        results = []
        
        with open(candidates_path, 'r', encoding='utf-8') as f:
            candidates = [json.loads(line.strip()) for line in f if line.strip()]
        
        print(f"Loaded {len(candidates)} candidates")
        
        for candidate in tqdm(candidates, desc="Ranking candidates"):
            candidate_id = candidate.get('candidate_id', '')
            
            semantic_sim = self.semantic_matcher.compute_candidate_similarity(candidate)
            score = self.rank_candidate(candidate, semantic_sim)
            
            results.append({
                'candidate_id': candidate_id,
                'score': score,
                'candidate': candidate
            })
        
        results.sort(key=lambda x: (-x['score'], x['candidate_id']))
        
        print("Generating reasoning and writing output...")
        final_results = []
        
        prev_score = 1.0
        for rank, item in enumerate(results[:100], start=1):
            curr_score = min(prev_score, round(item['score'], 4))
            prev_score = curr_score - 0.0001
            reasoning = self.reasoning_generator.generate_reasoning(item['candidate'], curr_score)
            final_results.append({
                'candidate_id': item['candidate_id'],
                'rank': rank,
                'score': curr_score,
                'reasoning': reasoning[:150]
            })
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['candidate_id', 'rank', 'score', 'reasoning'])
            writer.writeheader()
            writer.writerows(final_results)
        
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.1f} seconds")
        
        return final_results

def main():
    pipeline = CandidateRankingPipeline()
    results = pipeline.run(
        candidates_path='candidates.jsonl',
        output_path='submission.csv'
    )
    print(f"Generated ranking for top {len(results)} candidates")

if __name__ == '__main__':
    main()