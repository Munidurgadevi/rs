"""
Evaluation Framework Module for Ranking Systems.
Implements NDCG, MRR, MAP computation and candidate evaluation scoring.
"""

import math
from typing import Dict, List, Tuple

EVALUATION_KEYWORDS = {
    'ndcg', 'mrr', 'map', 'precision', 'recall', 'offline', 'online',
    'a/b test', 'ab test', 'experiment', 'evaluation', 'benchmark',
    'metric', 'offline-to-online', 'correlation', 'significance',
    'relevance', 'judgment', 'labeled', 'ground truth'
}


class EvaluationFramework:
    """Evaluates candidates on their experience with ranking system evaluation."""

    def compute_ndcg(self, relevance_scores: List[float], k: int = 10) -> float:
        """Compute Normalized Discounted Cumulative Gain."""
        if not relevance_scores:
            return 0.0
        
        dcg = sum(
            (2 ** rel - 1) / math.log2(i + 2)
            for i, rel in enumerate(relevance_scores[:k])
        )
        
        ideal_scores = sorted(relevance_scores, reverse=True)[:k]
        idcg = sum(
            (2 ** rel - 1) / math.log2(i + 2)
            for i, rel in enumerate(ideal_scores)
        )
        
        return dcg / idcg if idcg > 0 else 0.0

    def compute_mrr(self, relevance_scores: List[float]) -> float:
        """Compute Mean Reciprocal Rank."""
        for i, rel in enumerate(relevance_scores):
            if rel > 0:
                return 1.0 / (i + 1)
        return 0.0

    def compute_map(self, relevance_scores: List[float]) -> float:
        """Compute Mean Average Precision."""
        if not relevance_scores:
            return 0.0
        
        num_relevant = sum(1 for rel in relevance_scores if rel > 0)
        if num_relevant == 0:
            return 0.0
        
        avg_precisions = []
        relevant_so_far = 0
        
        for i, rel in enumerate(relevance_scores):
            if rel > 0:
                relevant_so_far += 1
                precision = relevant_so_far / (i + 1)
                avg_precisions.append(precision)
        
        return sum(avg_precisions) / num_relevant

    def compute_offline_online_correlation(self, offline_metrics: List[float], 
                                          online_metrics: List[float]) -> float:
        """Compute correlation between offline and online metrics."""
        if len(offline_metrics) != len(online_metrics) or len(offline_metrics) < 2:
            return 0.0
        
        n = len(offline_metrics)
        mean_off = sum(offline_metrics) / n
        mean_on = sum(online_metrics) / n
        
        cov = sum((offline_metrics[i] - mean_off) * (online_metrics[i] - mean_on) for i in range(n))
        std_off = math.sqrt(sum((x - mean_off) ** 2 for x in offline_metrics))
        std_on = math.sqrt(sum((x - mean_on) ** 2 for x in online_metrics))
        
        if std_off == 0 or std_on == 0:
            return 0.0
        
        return cov / (std_off * std_on)

    def score_evaluation_framework_experience(self, candidate: Dict) -> float:
        """
        Score a candidate based on their demonstrated experience with
        evaluation frameworks for ranking systems.
        
        Returns a score from 0.0 to 1.0.
        """
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
        
        # Explicit mention of ranking metrics
        metric_mentions = sum(1 for kw in ['ndcg', 'mrr', 'map', 'precision@k', 'recall@k'] if kw in text)
        score += min(metric_mentions * 0.15, 0.3)
        
        # A/B testing and experimentation
        if 'a/b test' in text or 'ab test' in text:
            score += 0.15
        if 'experiment' in text or 'experimentation' in text:
            score += 0.1
        
        # Offline-online correlation
        if 'offline-to-online' in text or 'offline online correlation' in text:
            score += 0.15
        if 'offline' in text and 'online' in text:
            score += 0.1
        
        # Relevance labeling and human judgment
        if 'relevance' in text and 'judgment' in text:
            score += 0.1
        if 'human' in text and ('label' in text or 'judgment' in text):
            score += 0.1
        
        # Evaluation infrastructure
        if 'eval harness' in text or 'evaluation framework' in text or 'eval pipeline' in text:
            score += 0.15
        if 'benchmark' in text:
            score += 0.05
        
        # Metric-driven optimization
        if 'metric' in text and ('optimize' in text or 'improve' in text):
            score += 0.1
        
        return min(1.0, score)

    def compute_simulated_ranking_metrics(self, candidate: Dict) -> Dict[str, float]:
        """
        Simulate ranking metrics for a candidate based on their profile signals.
        In production, these would be computed from actual relevance judgments.
        """
        profile = candidate.get('profile', {})
        career = candidate.get('career_history', [])
        signals = candidate.get('redrob_signals', {})
        
        years = profile.get('years_of_experience', 0)
        has_ir = any(term in ' '.join(r.get('description', '') for r in career).lower()
                    for term in ['retrieval', 'search', 'ranking', 'recommendation', 'vector', 'embedding'])
        has_prod = any(term in ' '.join(r.get('description', '') for r in career).lower()
                      for term in ['production', 'deployed', 'shipped', 'live'])
        resp_rate = signals.get('recruiter_response_rate', 0.5)
        
        # Simulate NDCG@10 based on experience and production quality
        base_ndcg = 0.3
        if years >= 5:
            base_ndcg += 0.15
        if has_ir:
            base_ndcg += 0.2
        if has_prod:
            base_ndcg += 0.15
        if resp_rate >= 0.7:
            base_ndcg += 0.1
        ndcg = min(1.0, base_ndcg + (hash(candidate.get('candidate_id', '')) % 100) / 500)
        
        # Simulate MRR
        mrr = min(1.0, ndcg * 0.7 + 0.1)
        
        # Simulate MAP
        map_score = min(1.0, ndcg * 0.8 + 0.05)
        
        return {
            'ndcg@10': round(ndcg, 4),
            'mrr': round(mrr, 4),
            'map': round(map_score, 4)
        }
