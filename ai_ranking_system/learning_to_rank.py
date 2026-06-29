"""
Learning-to-Rank Module.
Implements LTR models (XGBoost-based or neural) for candidate ranking.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


class LearningToRank:
    """
    Learning-to-Rank model for candidate ranking.
    Uses gradient-boosted trees for production robustness.
    """

    def __init__(self, use_xgboost: bool = True):
        self.use_xgboost = use_xgboost
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'semantic_similarity', 'skill_match', 'production_ml', 
            'career_trajectory', 'behavioral_signals', 'availability',
            'experience_years', 'product_company_ratio', 'evaluation_score',
            'response_rate', 'notice_period_days', 'profile_completeness'
        ]
        
        try:
            import xgboost as xgb
            self.xgb = xgb
            self._xgboost_available = True
        except ImportError:
            self._xgboost_available = False
            self.use_xgboost = False

    def extract_features(self, candidate: Dict, semantic_sim: float = 0.0) -> np.ndarray:
        """Extract feature vector from candidate profile."""
        profile = candidate.get('profile', {})
        career = candidate.get('career_history', [])
        skills = candidate.get('skills', [])
        signals = candidate.get('redrob_signals', {})
        
        years = profile.get('years_of_experience', 0)
        
        # Skill match score
        skill_names = [s.get('name', '').lower() for s in skills]
        core_ai = sum(1 for s in skill_names if any(t in s for t in [
            'embedding', 'retrieval', 'ranking', 'vector', 'search', 'milvus',
            'pinecone', 'faiss', 'nlp', 'llm', 'rag', 'bge', 'e5',
            'learning-to-rank', 'recommender', 'recommendation'
        ]))
        skill_match = min(1.0, core_ai / 5.0)
        
        # Production ML score
        text = ' '.join([
            profile.get('summary', ''),
            ' '.join(r.get('description', '') for r in career)
        ]).lower()
        prod_keywords = ['production', 'deployed', 'live', 'shipping', 'real users', 'latency', 'throughput']
        prod_count = sum(1 for kw in prod_keywords if kw in text)
        prod_score = min(1.0, prod_count / 4.0)
        
        # Career trajectory
        unique_companies = len(set(r.get('company', '') for r in career))
        stability = sum(1 for r in career if r.get('duration_months', 0) >= 24)
        traj_score = max(0.0, min(1.0, 0.5 + stability * 0.2 - max(0, unique_companies - 2) * 0.1))
        
        # Behavioral signals
        resp_rate = signals.get('recruiter_response_rate', 0.5)
        behavior_score = 0.3 * min(1.0, resp_rate * 2) + 0.3 * (signals.get('profile_completeness_score', 0) / 100)
        
        # Availability
        avail_score = 0.5
        if signals.get('open_to_work_flag', False):
            avail_score += 0.2
        notice = signals.get('notice_period_days', 90)
        if notice <= 30:
            avail_score += 0.15
        elif notice <= 60:
            avail_score += 0.1
        
        # Product company ratio
        total_months = sum(r.get('duration_months', 0) for r in career)
        prod_months = sum(r.get('duration_months', 0) for r in career 
                         if r.get('company', '').lower() not in {
                             'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini',
                             'mindtree', 'tech mahindra', 'hcl', 'lti', 'persistent'
                         })
        prod_ratio = prod_months / total_months if total_months > 0 else 0.5
        
        # Evaluation score (from evaluation_framework module)
        from .evaluation_framework import EvaluationFramework
        eval_fw = EvaluationFramework()
        eval_score = eval_fw.score_evaluation_framework_experience(candidate)
        
        profile_completeness = signals.get('profile_completeness_score', 0) / 100.0
        
        features = np.array([[
            semantic_sim,
            skill_match,
            prod_score,
            traj_score,
            behavior_score,
            avail_score,
            years,
            prod_ratio,
            eval_score,
            resp_rate,
            notice,
            profile_completeness
        ]])
        
        return features

    def train(self, candidates: List[Dict], labels: List[float], semantic_sims: List[float]) -> None:
        """Train the LTR model on candidate features and relevance labels."""
        X = np.vstack([
            self.extract_features(c, sim).flatten()
            for c, sim in zip(candidates, semantic_sims)
        ])
        y = np.array(labels)
        
        X_scaled = self.scaler.fit_transform(X)
        
        if self._xgboost_available and self.use_xgboost:
            self.model = self.xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                objective='reg:squarederror',
                random_state=42
            )
            self.model.fit(X_scaled, y)
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
            self.model.fit(X_scaled, y)

    def predict(self, candidate: Dict, semantic_sim: float = 0.0) -> float:
        """Predict ranking score for a candidate."""
        if self.model is None:
            return 0.0
        
        features = self.extract_features(candidate, semantic_sim)
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        return float(np.clip(prediction, 0.0, 1.0))

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model."""
        if self.model is None:
            return {}
        
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances.tolist()))
