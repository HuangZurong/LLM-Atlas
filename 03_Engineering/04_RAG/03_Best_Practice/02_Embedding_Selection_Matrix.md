# Embedding Selection Matrix

*Prerequisite: [../01_Theory/01_Architecture.md](../01_Theory/01_Architecture.md).*

---

Choosing the right embedding model is one of the highest-leverage decisions in a RAG system. A wrong choice here cannot be compensated by better chunking, reranking, or prompting.

## 1. The Selection Framework

### 1.1 Decision Criteria (Ranked by Impact)

| Priority | Criterion | Why It Matters |
| :--- | :--- | :--- |
| **P0** | **Language Coverage** | A model not trained on your language will fail silently — high cosine scores, wrong results. |
| **P1** | **MTEB/BEIR Benchmark Rank** | The closest proxy to real-world retrieval quality. Focus on the **Retrieval** subset, not overall. |
| **P2** | **Max Token Length** | If your chunks exceed the model's limit, trailing content is silently truncated. |
| **P3** | **Dimensionality** | Directly impacts vector DB storage cost and search latency at scale. |
| **P4** | **Inference Cost & Latency** | Matters for online embedding (query-time), less for offline indexing. |
| **P5** | **Matryoshka Support** | Allows truncating dimensions (e.g., 1024→256) with graceful degradation — critical for cost control. |

### 1.2 The Cardinal Rule

> **Never choose an embedding model based on blog posts or marketing. Always run a retrieval benchmark on YOUR data.**
>
> Public benchmarks (MTEB) test on academic datasets. Your domain (legal, medical, e-commerce) may behave very differently.

## 2. Model Comparison Matrix (2024-2025)

### 2.1 Proprietary Models

| Model | Provider | Dims | Max Tokens | Matryoshka | Multilingual | MTEB Retrieval Avg | Cost (per 1M tokens) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `text-embedding-3-large` | OpenAI | 3072 | 8191 | Yes (256-3072) | Good | ~56 | $0.13 |
| `text-embedding-3-small` | OpenAI | 1536 | 8191 | Yes (256-1536) | Good | ~51 | $0.02 |
| `voyage-3-large` | Voyage AI | 1024 | 32000 | No | Good | ~58 | $0.18 |
| `voyage-3-lite` | Voyage AI | 512 | 32000 | No | Good | ~52 | $0.02 |
| `embed-v4.0` | Cohere | 1024 | 512 | No | 100+ langs | ~55 | $0.10 |
| Gemini Embedding | Google | 768 | 2048 | No | Good | ~54 | Free (limited) |

### 2.2 Open-Source Models

| Model | Dims | Max Tokens | Matryoshka | Multilingual | MTEB Retrieval Avg | Params |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `bge-m3` (BAAI) | 1024 | 8192 | No | 100+ langs | ~55 | 568M |
| `bge-en-icl` (BAAI) | 4096 | 32768 | No | English | ~60 | 7B |
| `gte-Qwen2-7B-instruct` | 3584 | 32768 | No | Multilingual | ~60 | 7B |
| `jina-embeddings-v3` | 1024 | 8192 | Yes (64-1024) | 89 langs | ~57 | 570M |
| `nomic-embed-text-v1.5` | 768 | 8192 | Yes (64-768) | English | ~53 | 137M |
| `e5-mistral-7b-instruct` | 4096 | 32768 | No | English | ~57 | 7B |
| `multilingual-e5-large-instruct` | 1024 | 514 | No | 100+ langs | ~52 | 560M |

### 2.3 Specialized / Domain Models

| Model | Domain | Why Use It |
| :--- | :--- | :--- |
| `PubMedBERT` embeddings | Biomedical | Trained on PubMed/PMC corpus; outperforms general models on clinical text. |
| `Legal-BERT` embeddings | Legal | Better handling of legal terminology and citation patterns. |
| `CodeSage` / `StarEncoder` | Code | Trained on code corpora; understands function signatures, variable names. |

**Takeaway**: Domain-specific models beat general-purpose models by 10-20% on in-domain retrieval — but only if your corpus is purely in that domain.

## 3. Decision Trees

### 3.1 Quick Selection

```
Start
  ├─ Budget constrained? ──Yes──> text-embedding-3-small (cheapest, decent quality)
  │
  ├─ Multilingual required? ──Yes──> bge-m3 (open) or Cohere embed-v4 (API)
  │
  ├─ Long documents (>8K tokens)? ──Yes──> voyage-3-large or jina-v3
  │
  ├─ Self-hosted required? ──Yes──> bge-m3 (balanced) or nomic-v1.5 (lightweight)
  │
  ├─ Maximum quality (English)? ──Yes──> gte-Qwen2-7B or bge-en-icl (if GPU available)
  │
  └─ Default recommendation ──> text-embedding-3-large (Matryoshka + good quality)
```

### 3.2 Cost vs. Quality Trade-off

```
Quality ▲
        │  ★ gte-Qwen2-7B (self-hosted GPU cost)
        │  ★ bge-en-icl
        │  ★ voyage-3-large
        │  ★ jina-v3        ★ text-embedding-3-large
        │  ★ bge-m3
        │
        │          ★ text-embedding-3-small
        │  ★ nomic-v1.5
        │
        └──────────────────────────────────────► Cost per 1M tokens
          $0 (self-host)    $0.02    $0.10    $0.18
```

## 4. Critical Engineering Considerations

### 4.1 Matryoshka Embeddings (MRL)

Models supporting Matryoshka Representation Learning allow you to **truncate** the embedding vector to a shorter dimension without re-training.

**Production Pattern**:
- Index at **full dimensions** (e.g., 3072) for archival.
- Search at **reduced dimensions** (e.g., 256) for speed.
- Re-rank top candidates using full-dimension similarity.

**Measured Impact** (text-embedding-3-large):
| Dimensions | MTEB Retrieval | Storage (1M docs) | Search Latency |
| :--- | :--- | :--- | :--- |
| 3072 | 56.0 | ~12 GB | Baseline |
| 1024 | 55.2 | ~4 GB | ~3x faster |
| 256 | 52.1 | ~1 GB | ~12x faster |

### 4.2 Instruction-Tuned Embeddings

Modern models (BGE, E5, GTE) use **task-specific prefixes** to differentiate query vs. document embeddings:

```python
# BGE-M3 style
query_text = "Represent this sentence for searching relevant passages: " + user_query
doc_text = raw_chunk  # No prefix for documents

# E5 style
query_text = "query: " + user_query
doc_text = "passage: " + raw_chunk
```

**Forgetting the prefix is a silent killer** — retrieval quality drops 5-15% with no error message.

### 4.3 Embedding Model Migration

Changing your embedding model means **re-indexing your entire corpus**. Plan for this:

1. **Dual-Write Period**: Index new documents with both old and new models.
2. **Background Re-index**: Batch re-embed existing documents with the new model.
3. **Shadow Testing**: Route a % of queries to the new index, compare metrics.
4. **Cutover**: Switch traffic once RAGAS metrics confirm improvement.
5. **Cleanup**: Drop the old index after a rollback window.

**Rule of Thumb**: Budget 2-4 weeks for a full embedding migration in production. Never do it without a Golden Dataset to validate.

## 5. Anti-Patterns

| Anti-Pattern | Why It Fails | Fix |
| :--- | :--- | :--- |
| Using the same model for query and document without checking if it's asymmetric | Asymmetric models (BGE, E5) expect different prefixes for queries vs. docs | Read the model card; apply correct prefixes |
| Choosing by MTEB **overall** score | Overall includes classification, clustering, etc. — irrelevant for RAG | Filter MTEB to **Retrieval** tasks only |
| Embedding chunks longer than `max_tokens` | Excess tokens are silently truncated — you lose information | Enforce `chunk_size < model_max_tokens` in your pipeline |
| Using 7B embedding models without GPU | CPU inference is 50-100x slower; query latency becomes unacceptable | Use smaller models (137M-570M) for CPU, or provision GPU |
| Never re-evaluating after 6 months | The embedding landscape evolves fast; today's best is tomorrow's baseline | Schedule quarterly evaluations against your Golden Dataset |
