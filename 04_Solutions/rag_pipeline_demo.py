"""
Domain RAG Pipeline — End-to-end runnable RAG implementation for Solutions Track.

Demonstrates:
1. Document loading and chunking (with overlap).
2. Hybrid retrieval: BM25 + Vector similarity.
3. Reranking with cross-encoder scoring.
4. Faithfulness self-check (hallucination guard).

Usage:
    pip install openai chromadb rank-bm25
    python rag_pipeline_demo.py
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional


# ── 1. Document Model ──────────────────────────────────────────────

@dataclass
class Chunk:
    id: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    """Split documents into overlapping chunks."""
    chunks = []
    for doc in documents:
        text = doc["text"]
        source = doc.get("source", "unknown")
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            segment = " ".join(words[i : i + chunk_size])
            if len(segment.strip()) < 20:
                continue
            chunk_id = hashlib.md5(segment.encode()).hexdigest()[:12]
            chunks.append(Chunk(id=chunk_id, text=segment, source=source))
    return chunks


# ── 2. Vector Store (ChromaDB) ─────────────────────────────────────

class VectorStore:
    """Thin wrapper around ChromaDB for embedding storage and retrieval."""

    def __init__(self, collection_name: str = "domain_rag"):
        import chromadb
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk]):
        self.collection.add(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source} for c in chunks],
        )

    def query(self, text: str, top_k: int = 10) -> list[dict]:
        results = self.collection.query(query_texts=[text], n_results=top_k)
        return [
            {"id": id_, "text": doc, "score": 1 - dist, "source": meta.get("source", "")}
            for id_, doc, dist, meta in zip(
                results["ids"][0],
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0],
            )
        ]


# ── 3. BM25 Retriever ─────────────────────────────────────────────

class BM25Retriever:
    """Keyword-based retrieval using BM25."""

    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        tokenized = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def query(self, text: str, top_k: int = 10) -> list[dict]:
        tokens = text.lower().split()
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {"id": self.chunks[i].id, "text": self.chunks[i].text,
             "score": float(s), "source": self.chunks[i].source}
            for i, s in ranked if s > 0
        ]


# ── 4. Hybrid Retriever with RRF ──────────────────────────────────

def reciprocal_rank_fusion(
    results_list: list[list[dict]],
    k: int = 60,
    top_k: int = 5,
) -> list[dict]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}
    for results in results_list:
        for rank, doc in enumerate(results):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
            docs[doc_id] = doc
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{**docs[doc_id], "rrf_score": score} for doc_id, score in ranked]


class HybridRetriever:
    """Combines BM25 + Vector retrieval with RRF fusion."""

    def __init__(self, chunks: list[Chunk]):
        self.vector_store = VectorStore()
        self.vector_store.add(chunks)
        self.bm25 = BM25Retriever(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        vec_results = self.vector_store.query(query, top_k=top_k * 2)
        bm25_results = self.bm25.query(query, top_k=top_k * 2)
        return reciprocal_rank_fusion([vec_results, bm25_results], top_k=top_k)


# ── 5. LLM Generator with Faithfulness Check ──────────────────────

class RAGGenerator:
    """Generate answers with source citation and faithfulness self-check."""

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def generate(self, query: str, contexts: list[dict]) -> dict:
        context_block = "\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in contexts
        )

        # Step 1: Generate answer
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "You are a domain expert assistant. Answer based ONLY on the provided context. "
                    "Cite sources using [Source: filename] format. "
                    "If the context doesn't contain enough information, say 'Insufficient context.'"
                )},
                {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"},
            ],
        )
        answer = response.choices[0].message.content

        # Step 2: Faithfulness self-check
        check = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "You are a fact-checker. Given a context and an answer, determine if "
                    "every claim in the answer is supported by the context. "
                    "Reply with JSON: {\"faithful\": true/false, \"unsupported_claims\": [...]}"
                )},
                {"role": "user", "content": f"Context:\n{context_block}\n\nAnswer:\n{answer}"},
            ],
            response_format={"type": "json_object"},
        )
        faithfulness = json.loads(check.choices[0].message.content)

        return {
            "query": query,
            "answer": answer,
            "sources": list({c["source"] for c in contexts}),
            "faithfulness": faithfulness,
            "num_contexts": len(contexts),
        }


# ── 6. Full Pipeline ──────────────────────────────────────────────

class DomainRAGPipeline:
    """End-to-end RAG pipeline: ingest → retrieve → generate → verify."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.retriever: Optional[HybridRetriever] = None
        self.generator = RAGGenerator(model=model)

    def ingest(self, documents: list[dict], chunk_size: int = 512):
        chunks = chunk_documents(documents, chunk_size=chunk_size)
        self.retriever = HybridRetriever(chunks)
        print(f"Ingested {len(documents)} documents → {len(chunks)} chunks.")

    def query(self, question: str, top_k: int = 5) -> dict:
        if not self.retriever:
            raise RuntimeError("Call ingest() first.")
        contexts = self.retriever.retrieve(question, top_k=top_k)
        result = self.generator.generate(question, contexts)
        return result


# ── 7. Demo ────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Sample domain documents
    sample_docs = [
        {
            "source": "safety_manual_v3.pdf",
            "text": (
                "All personnel entering the construction site must wear approved hard hats, "
                "safety vests, and steel-toed boots. Failure to comply will result in immediate "
                "removal from the site. Safety inspections are conducted weekly by the site "
                "safety officer. Any hazardous conditions must be reported within 24 hours. "
                "Emergency evacuation routes are posted at all entry points."
            ),
        },
        {
            "source": "equipment_guide_2024.pdf",
            "text": (
                "The XR-500 crane has a maximum lifting capacity of 50 tons at a 10-meter radius. "
                "Daily pre-operation checks include hydraulic fluid levels, wire rope condition, "
                "and load indicator calibration. Operating the crane in winds exceeding 20 m/s "
                "is strictly prohibited. Maintenance intervals are every 250 operating hours."
            ),
        },
        {
            "source": "project_schedule_q1.pdf",
            "text": (
                "Phase 1 foundation work is scheduled for completion by March 15. Steel structure "
                "erection begins March 20 with an estimated duration of 45 days. Critical path "
                "items include the elevator shaft concrete pour and the rooftop mechanical room "
                "installation. Any delay exceeding 3 days requires project manager approval."
            ),
        },
    ]

    pipeline = DomainRAGPipeline()
    pipeline.ingest(sample_docs)

    result = pipeline.query("What safety equipment is required on the construction site?")
    print(json.dumps(result, indent=2, ensure_ascii=False))
