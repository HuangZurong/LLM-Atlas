import asyncio
from typing import List, Dict
from openai import AsyncOpenAI

"""
Industrial Pattern: Query Routing for RAG
-----------------------------------------
Not all queries should hit the same retrieval path.
Production RAG systems use a Router to classify the query
and dispatch it to the optimal retrieval strategy.

Routing strategies used in production:
1. Semantic Router: Embedding-based classification (fast, no LLM call).
2. LLM Router: Use a cheap model to classify intent (flexible, slower).
3. Rule-based Router: Regex/keyword matching (deterministic, limited).
"""

# ───────────────────────────────────────────────────────────────────────────
# Strategy 1: LLM-based Router (Most Flexible)
# ───────────────────────────────────────────────────────────────────────────

ROUTER_PROMPT = """
Classify the user query into ONE of these retrieval strategies:
- "vector_search": General semantic questions about concepts or topics.
- "keyword_search": Queries containing specific IDs, codes, or exact names.
- "graph_search": Questions requiring multi-hop reasoning across entities.
- "no_retrieval": Simple greetings or questions the LLM can answer directly.

Output ONLY the strategy name, nothing else.
"""

class LLMQueryRouter:
    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model

    async def route(self, query: str) -> str:
        res = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": query}
            ],
            max_tokens=20,
            temperature=0
        )
        return res.choices[0].message.content.strip().lower()

# ───────────────────────────────────────────────────────────────────────────
# Strategy 2: Semantic Router (No LLM call, Embedding-based)
# ───────────────────────────────────────────────────────────────────────────

import numpy as np

class SemanticRouter:
    """
    Pre-compute embeddings for route descriptions.
    At runtime, embed the query and find the closest route.
    Used by: semantic-router library, production chatbots.
    """
    def __init__(self):
        self.routes = {
            "vector_search": "Find information about concepts, topics, or general knowledge.",
            "keyword_search": "Look up a specific product ID, error code, or exact name.",
            "graph_search": "Answer a question that connects multiple entities or requires reasoning across documents.",
            "no_retrieval": "Simple greeting, small talk, or a question that doesn't need external data."
        }
        # In production: pre-compute embeddings via OpenAI/BGE
        self.route_embeddings = {k: np.random.rand(1536) for k in self.routes}

    def route(self, query_embedding: np.ndarray) -> str:
        best_route, best_score = "", -1
        for name, emb in self.route_embeddings.items():
            score = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            if score > best_score:
                best_score = score
                best_route = name
        return best_route

# ───────────────────────────────────────────────────────────────────────────
# Dispatcher: Route -> Execute
# ───────────────────────────────────────────────────────────────────────────

async def dispatch(route: str, query: str):
    """In production, each route calls a different retrieval backend."""
    handlers = {
        "vector_search": lambda q: f"[Vector DB] Searching for: {q}",
        "keyword_search": lambda q: f"[BM25/Elastic] Exact match for: {q}",
        "graph_search": lambda q: f"[Neo4j/Graph] Multi-hop traversal for: {q}",
        "no_retrieval": lambda q: f"[Direct LLM] No retrieval needed for: {q}",
    }
    handler = handlers.get(route, handlers["vector_search"])
    return handler(query)

async def main():
    client = AsyncOpenAI(api_key="sk-...")
    router = LLMQueryRouter(client)

    queries = [
        "What is retrieval augmented generation?",
        "Find product SKU-12345",
        "How does the warranty relate to the recall notice?",
        "Hello!"
    ]

    for q in queries:
        # route = await router.route(q)
        # result = await dispatch(route, q)
        # print(f"Query: {q}\n  Route: {route}\n  Result: {result}\n")
        pass

    print("Query Router (LLM + Semantic) ready.")

if __name__ == "__main__":
    asyncio.run(main())
