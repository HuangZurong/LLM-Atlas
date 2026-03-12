import numpy as np
from typing import List, Dict, Optional
from openai import OpenAI

"""
Industrial Pattern: Vector-based Long-Term Memory
-------------------------------------------------
For cross-session memory (user preferences, past decisions),
we store conversation turns as embeddings and retrieve the most
relevant ones when needed.

Production stack: OpenAI Embeddings + Qdrant/Pinecone/Chroma.
This example uses in-memory numpy for demonstration.
"""

class VectorMemoryStore:
    def __init__(self, client: OpenAI, embedding_model: str = "text-embedding-3-small"):
        self.client = client
        self.embedding_model = embedding_model
        self.memories: List[Dict] = []  # {text, embedding, metadata}

    def _embed(self, text: str) -> np.ndarray:
        """In production, call the embedding API. Mock here for demo."""
        # res = self.client.embeddings.create(model=self.embedding_model, input=text)
        # return np.array(res.data[0].embedding)
        np.random.seed(hash(text) % 2**32)
        return np.random.rand(1536)

    def store(self, text: str, metadata: Optional[Dict] = None):
        emb = self._embed(text)
        self.memories.append({
            "text": text,
            "embedding": emb,
            "metadata": metadata or {}
        })

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        query_emb = self._embed(query)
        scored = []
        for mem in self.memories:
            sim = np.dot(query_emb, mem['embedding']) / (
                np.linalg.norm(query_emb) * np.linalg.norm(mem['embedding'])
            )
            scored.append((sim, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"text": m['text'], "score": round(s, 4), **m['metadata']}
                for s, m in scored[:top_k]]

    def to_context_string(self, query: str, top_k: int = 3) -> str:
        """Format retrieved memories for injection into the prompt."""
        results = self.retrieve(query, top_k)
        if not results:
            return ""
        lines = [f"- {r['text']} (relevance: {r['score']})" for r in results]
        return "[Retrieved Memories]:\n" + "\n".join(lines)

if __name__ == "__main__":
    client = OpenAI(api_key="sk-...")
    store = VectorMemoryStore(client)

    # Simulate storing past interactions
    store.store("User prefers Python over JavaScript.", {"session": "2026-01-15"})
    store.store("We decided on 512-token chunks with 50-token overlap.", {"session": "2026-01-20"})
    store.store("Budget for the project is $50K.", {"session": "2026-02-01"})

    # Retrieve relevant memories
    context = store.to_context_string("What chunking strategy did we decide on?")
    print(context)
