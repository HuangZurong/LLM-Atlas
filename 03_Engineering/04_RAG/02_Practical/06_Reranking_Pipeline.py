import asyncio
from typing import List, Dict
from openai import AsyncOpenAI

"""
Industrial Pattern: Reranking Pipeline
--------------------------------------
Vector search returns "approximately similar" results.
In production, a Reranker (Cross-Encoder) performs a deep
pairwise comparison between the query and each candidate
to produce a precise relevance score.

Stack: Cohere Rerank API / BGE-Reranker / Jina Reranker
"""

class RerankingPipeline:
    def __init__(self, client: AsyncOpenAI, rerank_model: str = "gpt-4o-mini"):
        self.client = client
        self.rerank_model = rerank_model

    async def rerank(self, query: str, candidates: List[str], top_k: int = 3) -> List[Dict]:
        """
        Uses an LLM as a reranker (production alternative: Cohere/BGE Cross-Encoder).
        Scores each candidate's relevance to the query on a 0-10 scale.
        """
        prompt = f"""Score each document's relevance to the query on a scale of 0-10.
Query: "{query}"

Documents:
{chr(10).join(f'{i+1}. {c}' for i, c in enumerate(candidates))}

Return ONLY a JSON array of scores, e.g. [8, 3, 9, ...]"""

        res = await self.client.chat.completions.create(
            model=self.rerank_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )

        import json
        scores_raw = json.loads(res.choices[0].message.content)
        # Handle both {"scores": [...]} and direct [...] formats
        scores = scores_raw if isinstance(scores_raw, list) else scores_raw.get("scores", [])

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [{"text": text, "score": score} for text, score in ranked[:top_k]]

# ───────────────────────────────────────────────────────────────────────────
# Full Retrieval Pipeline: Search -> Filter -> Rerank
# ───────────────────────────────────────────────────────────────────────────

class RetrievalPipeline:
    """
    The complete retrieval flow used in production RAG:
    1. Initial Retrieval (top-50 from vector DB)
    2. Metadata Filtering (hard constraints)
    3. Reranking (top-5 via Cross-Encoder)
    4. Context Assembly (format for LLM prompt)
    """
    def __init__(self, reranker: RerankingPipeline):
        self.reranker = reranker

    async def execute(self, query: str, initial_results: List[Dict], top_k: int = 3) -> str:
        # Step 1: Extract texts
        texts = [r['text'] for r in initial_results]

        # Step 2: Rerank
        reranked = await self.reranker.rerank(query, texts, top_k=top_k)

        # Step 3: Assemble context for LLM
        context_parts = [f"[Source {i+1}, Relevance: {r['score']}/10]\n{r['text']}"
                         for i, r in enumerate(reranked)]
        return "\n\n".join(context_parts)

async def main():
    client = AsyncOpenAI(api_key="sk-...")
    reranker = RerankingPipeline(client)
    pipeline = RetrievalPipeline(reranker)

    # Simulated vector search results (top-5 from DB)
    mock_results = [
        {"text": "The warranty covers manufacturing defects for 2 years."},
        {"text": "Product recall notice issued for batch #2024-Q3."},
        {"text": "Customer satisfaction survey results from 2023."},
        {"text": "Warranty claims must be filed within 30 days of discovery."},
        {"text": "Company holiday schedule for 2024."},
    ]

    # context = await pipeline.execute("Is my warranty affected by the recall?", mock_results)
    # print(context)
    print("Reranking Pipeline (Search -> Filter -> Rerank -> Assemble) ready.")

if __name__ == "__main__":
    asyncio.run(main())
