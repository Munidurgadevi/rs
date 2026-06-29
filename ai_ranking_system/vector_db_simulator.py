"""
Vector Database Simulator for Hybrid Search.
Implements dense + sparse hybrid retrieval using FAISS and TF-IDF.
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class VectorDBSimulator:
    """
    Simulates vector database operations for production retrieval systems.
    Supports dense embeddings + sparse BM25-like hybrid search.
    """

    def __init__(self, use_faiss: bool = True):
        self.use_faiss = use_faiss
        self.faiss_index = None
        self.embeddings = None
        self.ids = []
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        self.tfidf_matrix = None
        self.documents = []
        
        try:
            import faiss
            self.faiss = faiss
            self._faiss_available = True
        except ImportError:
            self._faiss_available = False
            self.use_faiss = False

    def build_index(self, documents: List[Dict], embedding_fn=None) -> None:
        """Build hybrid search index from documents."""
        self.documents = documents
        self.ids = [doc.get('id', str(i)) for i, doc in enumerate(documents)]
        
        texts = [doc.get('text', '') for doc in documents]
        
        # Build sparse TF-IDF index
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # Build dense embedding index
        if embedding_fn is not None:
            self.embeddings = np.array([embedding_fn(text) for text in texts], dtype=np.float32)
            if self._faiss_available and self.use_faiss:
                dimension = self.embeddings.shape[1]
                self.faiss_index = self.faiss.IndexFlatIP(dimension)
                self.faiss_index.add(self.embeddings)

    def hybrid_search(self, query: str, embedding_fn=None, top_k: int = 10,
                     alpha: float = 0.5) -> List[Tuple[str, float]]:
        """
        Perform hybrid search combining sparse (BM25/TF-IDF) and dense (embedding) retrieval.
        
        alpha: weight for dense retrieval (1-alpha for sparse)
        """
        if not self.documents:
            return []
        
        # Sparse retrieval score
        query_tfidf = self.tfidf_vectorizer.transform([query])
        sparse_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
        
        # Dense retrieval score
        dense_scores = np.zeros(len(self.documents))
        if embedding_fn is not None and self.embeddings is not None:
            query_embedding = embedding_fn(query).reshape(1, -1).astype(np.float32)
            if self._faiss_available and self.faiss_index is not None:
                _, indices = self.faiss_index.search(query_embedding, len(self.documents))
                for rank, idx in enumerate(indices[0]):
                    dense_scores[idx] = 1.0 / (rank + 1)
            else:
                query_norm = query_embedding / np.linalg.norm(query_embedding)
                emb_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
                dense_scores = (query_norm @ emb_norm.T).flatten()
        
        # Combine scores
        combined_scores = alpha * dense_scores + (1 - alpha) * sparse_scores
        
        # Get top-k results
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        results = [(self.ids[i], float(combined_scores[i])) for i in top_indices]
        
        return results

    def refresh_index(self, new_documents: List[Dict], embedding_fn=None) -> None:
        """Simulate incremental index refresh for embedding drift handling."""
        self.build_index(new_documents, embedding_fn)

    def detect_embedding_drift(self, old_embeddings: np.ndarray, 
                               new_embeddings: np.ndarray) -> float:
        """Detect embedding drift between old and new embeddings."""
        if old_embeddings.shape != new_embeddings.shape:
            return 1.0
        
        drift = np.mean(np.linalg.norm(old_embeddings - new_embeddings, axis=1))
        return float(drift)
