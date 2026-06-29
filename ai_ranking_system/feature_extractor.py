"""
Feature Engineering Pipeline for candidate profiles.
Computes production ML experience, search/ranking experience, and product company signals.
"""

import re
from typing import Dict, List, Set
from collections import Counter
from datetime import datetime

class FeatureExtractor:
    def __init__(self):
        self.production_ml_keywords = {
            'deployed', 'production', 'real-users', 'real users', 'live', 'shipping',
            'microservice', 'api', 'scal', 'throughput', 'latency', 'replic', 'index',
            'refresh', 'regression', 'a/b test', 'online learning', 'monitoring',
            'serving', 'kubernetes', 'docker', 'bentoml', 'mlflow', 'kubeflow'
        }
        
        self.search_ranking_keywords = {
            'retrieval', 'search', 'ranking', 'recommendation', 'recommender', 
            'embedding', 'vector', 'semantic', 'bm25', 'hybrid search',
            'similarity', 'candidate matching', 'job matching', 'rank',
            'milvus', 'pinecone', 'weaviate', 'qdrant', 'faiss', 'elasticsearch', 
            'opensearch', 'sentence-transformers', 'bge', 'e5', 'rag', 'rerank',
            'cross-encoder', 'dense retrieval', 'sparse retrieval', 'indexing',
            'learning-to-rank', 'xgb', 'xgboost', 'lightgbm', 'lamdamart'
        }
        
        self.evaluation_keywords = {
            'ndcg', 'mrr', 'map', 'precision', 'recall', 'offline', 'online',
            'a/b test', 'ab test', 'experiment', 'evaluation', 'benchmark',
            'metric', 'offline-to-online', 'relevance', 'judgment', 'labeled'
        }
        
        self.python_indicators = {
            'python', 'pytorch', 'tensorflow', 'scikit-learn', 'sklearn', 'numpy',
            'pandas', 'flask', 'django', 'fastapi', 'bentoml', 'mlflow'
        }
        
        self.product_companies = {
            'ola', 'razorpay', 'flipkart', 'zomato', 'swiggy', 'paytm', 'phonepe',
            'cred', 'groww', 'policybazaar', 'nykaa', 'zoho', 'freshworks', 'hasura',
            'postman', 'browserstack', 'hotstar', 'altbalaji', 'sony liv',
            'pied piper', 'hooli', 'wayne enterprises', 'acme corp', 'globex inc',
            'initech', 'dunder mifflin', 'stark industries', 'dream11', 'meesho',
            'inmobi', 'vedantu', 'amazon', 'google', 'microsoft', 'uber', 'adobe',
            'apple', 'salesforce', 'ola', 'uber', 'repharse.ai', 'aganitha',
            'pharmeasy', 'freshworks', 'verloop.io'
        }
        
        self.consulting_firms = {
            'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini', 
            'mindtree', 'tech mahindra', 'lti', 'persistent', 'hcl'
        }
        
        self.ai_engineer_titles = {
            'ai engineer', 'ml engineer', 'machine learning engineer',
            'data scientist', 'applied scientist', 'research engineer', 'ai researcher',
            'nlp engineer', 'deep learning engineer', 'search engineer',
            'recommendation systems engineer', 'applied ml engineer'
        }

    def extract_production_ml_score(self, candidate: Dict) -> float:
        text = self._get_candidate_text(candidate).lower()
        
        production_matches = sum(1 for kw in self.production_ml_keywords if kw in text)
        search_matches = sum(1 for kw in self.search_ranking_keywords if kw in text)
        
        production_score = min(production_matches / 5.0, 1.0)
        search_score = min(search_matches / 8.0, 1.0)
        
        return 0.6 * production_score + 0.4 * search_score

    def extract_product_company_score(self, candidate: Dict) -> float:
        career_history = candidate.get('career_history', [])
        product_experience = 0
        total_experience = 0
        
        for role in career_history:
            company = role.get('company', '').lower()
            duration = role.get('duration_months', 0)
            total_experience += duration
            
            if company not in self.consulting_firms:
                product_experience += duration
        
        if total_experience == 0:
            return 0.0
        
        return min(product_experience / total_experience, 1.0)

    def extract_search_ranking_score(self, candidate: Dict) -> float:
        text = self._get_candidate_text(candidate).lower()
        
        search_matches = sum(1 for kw in self.search_ranking_keywords if kw in text)
        eval_matches = sum(1 for kw in self.evaluation_keywords if kw in text)
        
        search_score = min(search_matches / 6.0, 1.0)
        eval_score = min(eval_matches / 3.0, 1.0)
        
        return 0.7 * search_score + 0.3 * eval_score

    def extract_python_score(self, candidate: Dict) -> float:
        text = self._get_candidate_text(candidate).lower()
        python_matches = sum(1 for kw in self.python_indicators if kw in text)
        return min(python_matches / 8.0, 1.0)

    def extract_experience_score(self, candidate: Dict) -> float:
        """
        Extract experience score with bell curve centered at 5-9 years per JD.
        JD: 5-9 years preferred. Flexible for strong candidates outside band.
        """
        years = candidate.get('profile', {}).get('years_of_experience', 0)
        
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

    def extract_career_trajectory_score(self, candidate: Dict) -> float:
        career_history = candidate.get('career_history', [])
        if len(career_history) < 2:
            return 0.3
        
        companies = [r.get('company', '') for r in career_history]
        unique_companies = len(set(companies))
        
        title_progression = 0
        for role in career_history:
            title = role.get('title', '').lower()
            if any(t in title for t in ['senior', 'lead', 'principal', 'staff', 'architect']):
                title_progression += 1
        
        # Penalize title-chasing (switching every ~1.5 years for titles)
        avg_tenure = sum(r.get('duration_months', 0) for r in career_history) / len(career_history)
        title_chase_penalty = 0.0
        if unique_companies >= 3 and avg_tenure < 18:
            title_chase_penalty = 0.2
        
        job_hopping_penalty = max(0, (unique_companies - 2) * 0.05)
        
        stability_bonus = 0
        for i in range(len(career_history) - 1):
            if career_history[i].get('duration_months', 0) >= 24:
                stability_bonus += 0.1
        
        return max(0.2, min(1.0, title_progression * 0.2 + stability_bonus - job_hopping_penalty - title_chase_penalty))

    def extract_company_quality_score(self, candidate: Dict) -> float:
        """Score based on quality of companies worked at (product vs consulting)."""
        career_history = candidate.get('career_history', [])
        if not career_history:
            return 0.0
        
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
        total_months = 0
        
        for role in career_history:
            company = role.get('company', '').lower()
            duration = role.get('duration_months', 0)
            total_months += duration
            
            if company in strong_product:
                score += duration * 1.0
            elif company in good_product:
                score += duration * 0.8
            elif company in self.product_companies:
                score += duration * 0.6
            elif company in self.consulting_firms:
                score += duration * 0.1
            else:
                score += duration * 0.3
        
        return min(1.0, score / total_months) if total_months > 0 else 0.0

    def _get_candidate_text(self, candidate: Dict) -> str:
        text_parts = []
        
        profile = candidate.get('profile', {})
        text_parts.append(profile.get('headline', ''))
        text_parts.append(profile.get('summary', ''))
        
        for role in candidate.get('career_history', []):
            text_parts.append(role.get('title', ''))
            text_parts.append(role.get('description', ''))
        
        for skill in candidate.get('skills', []):
            text_parts.append(skill.get('name', ''))
        
        return ' '.join(text_parts)