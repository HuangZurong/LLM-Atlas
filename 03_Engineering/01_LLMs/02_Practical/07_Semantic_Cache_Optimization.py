import numpy as np
from typing import Optional, Dict

"""
Industrial Best Practice: Semantic Caching
-----------------------------------------
Why?
1. Save Cost: Identical/Similar questions don't hit the LLM.
2. Speed: 10ms vs 2000ms.

How?
Instead of exact string match (Redis), we use Vector Embeddings to find
"Semantically Similar" questions.
"""

class SemanticCache:
    def __init__(self, threshold: float = 0.95):
        self.cache: Dict[str, str] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.threshold = threshold

    def _get_embedding(self, text: str) -> np.ndarray:
        # Real world: client.embeddings.create(model="text-embedding-3-small")
        # Mocking embedding as a simple normalized vector
        return np.random.rand(1536)

    def query(self, text: str) -> Optional[str]:
        """Check if a similar question exists in cache."""
        target_emb = self._get_embedding(text)

        for cached_text, emb in self.embeddings.items():
            # Cosine Similarity
            sim = np.dot(target_emb, emb) / (np.linalg.norm(target_emb) * np.linalg.norm(emb))
            if sim > self.threshold:
                print(f"[Cache Hit] Similarity: {sim:.4f}")
                return self.cache[cached_text]

        return None

    def update(self, text: str, response: str):
        emb = self._get_embedding(text)
        self.cache[text] = response
        self.embeddings[text] = emb

if __name__ == "__main__":
    cache = SemanticCache(threshold=0.9)

    # 1. First query (Miss)
    q1 = "How to center a div?"
    if not cache.query(q1):
        ans1 = "Use display: flex and justify-content: center."
        cache.update(q1, ans1)

    # 2. Similar query (Hit)
    q2 = "Way to center a div element?"
    hit = cache.query(q2)
    if hit:
        print(f"Cached Answer: {hit}")

    print("\nSemantic Cache mechanism (Vector-based) ready.")
