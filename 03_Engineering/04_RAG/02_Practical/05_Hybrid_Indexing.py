from typing import List, Dict
import numpy as np

"""
Industrial Pattern: Hybrid Indexing (Dense + Sparse + Metadata)
--------------------------------------------------------------
Production RAG systems never rely on a single retrieval path.
The "100% Success Formula" combines:
1. Dense (Vector): Semantic intent matching.
2. Sparse (BM25): Exact keyword/ID matching.
3. Metadata Filtering: Hard constraints (date, category, version).

This module demonstrates the indexing side: how to prepare documents
for hybrid retrieval.
"""

class HybridIndex:
    """
    Simulates a hybrid index that stores both dense and sparse
    representations alongside structured metadata.
    In production: use Weaviate, Qdrant, or LanceDB which support
    hybrid search natively.
    """
    def __init__(self):
        self.documents: List[Dict] = []

    def add_document(self, text: str, metadata: Dict = None):
        doc = {
            "text": text,
            "dense_vector": self._mock_embed(text),
            "sparse_tokens": self._bm25_tokenize(text),
            "metadata": metadata or {}
        }
        self.documents.append(doc)

    def _mock_embed(self, text: str) -> np.ndarray:
        np.random.seed(hash(text) % 2**32)
        return np.random.rand(1536)

    def _bm25_tokenize(self, text: str) -> Dict[str, int]:
        """Simple term frequency for BM25-style sparse representation."""
        tokens = text.lower().split()
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        return tf

    def hybrid_search(self, query: str, filters: Dict = None, top_k: int = 5,
                      dense_weight: float = 0.7, sparse_weight: float = 0.3) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF) of dense and sparse results.
        RRF Score = sum(1 / (k + rank)) across all result sets.
        """
        query_vec = self._mock_embed(query)
        query_tokens = self._bm25_tokenize(query)

        # Step 1: Metadata hard filtering
        candidates = self.documents
        if filters:
            candidates = [d for d in candidates
                          if all(d['metadata'].get(k) == v for k, v in filters.items())]

        # Step 2: Dense scoring
        dense_scores = []
        for doc in candidates:
            sim = np.dot(query_vec, doc['dense_vector']) / (
                np.linalg.norm(query_vec) * np.linalg.norm(doc['dense_vector']))
            dense_scores.append((sim, doc))
        dense_scores.sort(key=lambda x: x[0], reverse=True)

        # Step 3: Sparse scoring (simplified BM25)
        sparse_scores = []
        for doc in candidates:
            score = sum(doc['sparse_tokens'].get(t, 0) for t in query_tokens)
            sparse_scores.append((score, doc))
        sparse_scores.sort(key=lambda x: x[0], reverse=True)

        # Step 4: RRF Fusion
        k = 60  # Standard RRF constant
        rrf_scores = {}
        for rank, (_, doc) in enumerate(dense_scores):
            doc_id = id(doc)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + dense_weight / (k + rank + 1)
        for rank, (_, doc) in enumerate(sparse_scores):
            doc_id = id(doc)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + sparse_weight / (k + rank + 1)

        # Sort by RRF score
        doc_map = {id(d): d for d in candidates}
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"text": doc_map[did]['text'], "rrf_score": round(score, 6)}
                for did, score in ranked[:top_k]]

if __name__ == "__main__":
    idx = HybridIndex()

    # Index documents with metadata
    idx.add_document("iPhone 15 Pro has a titanium frame and A17 chip.", {"brand": "Apple", "year": 2024})
    idx.add_document("iPhone 14 Pro features a stainless steel frame.", {"brand": "Apple", "year": 2023})
    idx.add_document("Galaxy S24 Ultra uses Snapdragon 8 Gen 3.", {"brand": "Samsung", "year": 2024})

    # Hybrid search with metadata filter
    results = idx.hybrid_search("titanium frame phone", filters={"brand": "Apple"}, top_k=2)
    for r in results:
        print(f"  [{r['rrf_score']}] {r['text']}")
