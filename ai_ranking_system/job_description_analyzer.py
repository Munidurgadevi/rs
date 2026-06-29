"""
Job Description Semantic Analyzer for Senior AI Engineer role.
Extracts key requirements and creates semantic embeddings.
"""

import re
from dataclasses import dataclass
from typing import List, Set, Dict

@dataclass
class JobRequirements:
    core_competencies: Set[str]
    preferred_skills: Set[str]
    disallowed_backgrounds: Set[str]
    location_preferences: Set[str]
    experience_range: tuple
    must_have_production: bool
    must_have_product_company: bool

class JobDescriptionAnalyzer:
    def __init__(self):
        self.job_text = """
        Senior AI Engineer - Founding Team at Redrob AI. We need someone with deep technical 
        depth in modern ML systems - embeddings, retrieval, ranking, LLMs, fine-tuning. 
        Must have production experience with embeddings-based retrieval systems and vector databases. 
        Strong Python required. Must have shipped end-to-end ranking, search, or recommendation 
        systems to real users at meaningful scale. Product-company experience required - no pure 
        research backgrounds. No consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini).
        Location: Pune/Noida preferred, open to Hyderabad, Mumbai, Delhi NCR, willing to relocate.
        Experience: 5-9 years preferred. Must be open to work and actively engaged.
        """
        
        self.prohibited_titles = {
            'marketing manager', 'sales', 'operations', 'project manager', 'hr manager', 
            'accountant', 'content writer', 'graphic designer', 'civil engineer', 'mechanical engineer',
            'customer support', 'business analyst'
        }
        
        self.consulting_firms = {
            'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini', 'mindtree',
            'tech mahindra', 'hcl', 'lti', 'persistent', 'cognizant'
        }
        
        self.core_ai_terms = {
            'embedding', 'retrieval', 'ranking', 'recommendation', 'search', 'vector', 'milvus', 
            'pinecone', 'weaviate', 'qdrant', 'faiss', 'elasticsearch', 'opensearch',
            'sentence-transformers', 'bge', 'e5', 'machine learning', 'ml', 'nlp', 'llm',
            'fine-tuning', 'lora', 'qlora', 'peft', 'pytorch', 'tensorflow', 'scikit-learn',
            'xgboost', 'learning-to-rank', 'ndcg', 'mrr', 'map', 'evaluation', 'rag',
            'semantic search', 'vector search', 'hybrid search'
        }
        
        self.industry_terms = {
            'product', 'saas', 'tech', 'software', 'marketplace', 'ai', 'ml', 'data science',
            'machine learning', 'nlp', 'computer vision', 'speech'
        }
        
        self.location_prefs = {
            'pune', 'noida', 'hyderabad', 'mumbai', 'delhi', 'ncr', 'india', 'bangalore'
        }

    def get_requirements(self) -> JobRequirements:
        return JobRequirements(
            core_competencies=self.core_ai_terms,
            preferred_skills={'distributed systems', 'inference optimization', 'open-source', 'hr-tech'},
            disallowed_backgrounds=self.prohibited_titles | self.consulting_firms,
            location_preferences=self.location_prefs,
            experience_range=(5, 9),
            must_have_production=True,
            must_have_product_company=True
        )

    def get_job_embedding_text(self) -> str:
        return ("Senior AI Engineer role requiring production ML systems experience. "
                "Must have built embeddings-based retrieval systems, vector databases, search infrastructure. "
                "Strong Python required. Must have shipped ranking/search/recommendation to real users. "
                "Product company experience mandatory. No research-only or consulting backgrounds. "
                "Located in or willing to relocate to Pune, Noida, Hyderabad, Mumbai, Delhi NCR.")