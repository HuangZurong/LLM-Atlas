# Production Architecture Decisions

*Prerequisite: [../01_Theory/02_Advanced_RAG.md](../01_Theory/02_Advanced_RAG.md).*

---

This document captures the key architectural decision points when designing a production RAG system. Each section presents the trade-off, a decision framework, and real-world guidance.

## 1. Retrieval Architecture Selection

### 1.1 Decision Matrix

| Architecture | When to Use | When NOT to Use | Real-World Example |
| :--- | :--- | :--- | :--- |
| **Vector-Only** | Homogeneous text corpus, semantic queries dominate | Queries contain IDs, codes, exact terms | Internal wiki Q&A |
| **Hybrid (Vector + BM25)** | Mixed query types, product catalogs, technical docs | Pure conversational queries with no keyword anchors | E-commerce search, API docs |
| **GraphRAG** | Multi-hop reasoning, entity-relationship questions | Simple factoid lookup, high-throughput low-latency | Compliance analysis, research synthesis |
| **Agentic RAG** | Heterogeneous data sources, queries need routing/planning | Latency-sensitive (<2s), simple single-source lookup | Enterprise knowledge assistant |

### 1.2 The 80/20 Rule

> **80% of production RAG systems should use Hybrid Search (Vector + BM25) with a Reranker.**
>
> Start here. Only move to GraphRAG or Agentic RAG when you have concrete evidence that Hybrid fails on your query distribution.

### 1.3 Architecture Progression Path

```
Stage 1: Vector-Only + Top-K
  → Problem: keyword queries fail (product IDs, error codes)

Stage 2: Hybrid Search (Vector + BM25 + RRF)
  → Problem: top-K still noisy, wrong chunk ranked #1

Stage 3: Hybrid + Cross-Encoder Reranker
  → Problem: multi-hop questions need info from multiple docs

Stage 4: Hybrid + Reranker + Query Decomposition
  → Problem: need to reason over entity relationships

Stage 5: GraphRAG or Agentic RAG
```

**Do not skip stages.** Each stage solves a specific failure mode. Jumping to Stage 5 without evidence wastes engineering effort and adds latency.

## 2. Vector Database Selection

### 2.1 Comparison Matrix

| Database | Type | Hybrid Search | Metadata Filtering | Managed Option | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pinecone** | Cloud-native | Yes (2024+) | Yes | Yes (only) | Teams wanting zero-ops |
| **Weaviate** | Self-host / Cloud | Yes (native) | Yes | Yes | Hybrid search-first architectures |
| **Qdrant** | Self-host / Cloud | Yes (sparse vectors) | Yes (rich filtering) | Yes | Complex filtering + high performance |
| **Milvus/Zilliz** | Self-host / Cloud | Yes | Yes | Yes (Zilliz) | Large-scale (billions of vectors) |
| **Chroma** | Embedded | No (vector only) | Basic | No | Prototyping, local dev |
| **pgvector** | PostgreSQL extension | Manual (with FTS) | Yes (SQL) | Via managed PG | Teams already on PostgreSQL |
| **LanceDB** | Embedded / Serverless | Yes | Yes | Serverless | Cost-sensitive, serverless-first |

### 2.2 Decision Framework

```
Start
  ├─ Prototype / <100K docs? ──Yes──> Chroma or LanceDB (embedded, zero setup)
  │
  ├─ Already on PostgreSQL? ──Yes──> pgvector (avoid new infra)
  │
  ├─ Need managed + zero-ops? ──Yes──> Pinecone or Zilliz Cloud
  │
  ├─ Hybrid search critical? ──Yes──> Weaviate or Qdrant
  │
  ├─ Billions of vectors? ──Yes──> Milvus / Zilliz
  │
  └─ Default ──> Qdrant (best balance of features, performance, flexibility)
```

### 2.3 The pgvector Trap

Many teams default to pgvector because "we already have Postgres." This works until:
- **>5M vectors**: HNSW index build time and memory become painful.
- **High QPS**: pgvector shares resources with your application DB; search latency spikes under load.
- **Hybrid search**: Combining FTS + vector in a single SQL query is possible but awkward and hard to tune.

**Guideline**: pgvector is excellent for <1M vectors with moderate QPS. Beyond that, use a dedicated vector DB.

## 3. Chunking Strategy Decisions

### 3.1 Strategy Selection

| Data Type | Recommended Strategy | Chunk Size | Overlap |
| :--- | :--- | :--- | :--- |
| **General text** (blogs, wiki) | Recursive Character | 512 tokens | 10-15% |
| **Technical docs** (API refs) | Markdown Header-based | Natural sections | None |
| **Legal / Medical** | Adaptive (similarity-based) | Dynamic | Dynamic |
| **Code** | AST-based (functions/classes) | Natural units | None |
| **Dense knowledge** (textbooks) | Proposition-based | Atomic facts | None |
| **Long narratives** (books) | RAPTOR (hierarchical) | Multi-level | N/A |

### 3.2 The Parent-Child Pattern (Production Standard)

Most production systems decouple the **search unit** from the **generation unit**:

```
Index:    Small child chunks (100-200 tokens) → high retrieval precision
Retrieve: Match on child chunks
Return:   Parent chunk (500-1000 tokens) → full context for LLM
```

**Why**: Small chunks embed cleanly (less noise), but LLMs need surrounding context to generate good answers. This pattern gives you both.

## 4. Online vs. Offline Architecture

### 4.1 Component Latency Budget

For a target of **<3 seconds** end-to-end:

```
┌──────────────────────────────────────────────────┐
│ Query Embedding          ~50ms                   │
│ Vector Search            ~30ms                   │
│ BM25 Search              ~20ms                   │
│ RRF Fusion               ~5ms                    │
│ Reranking (top-20)       ~200ms                  │
│ ─────────────────────────────────────────────     │
│ Retrieval Total          ~305ms                  │
│                                                  │
│ LLM Generation (streaming first token) ~500ms    │
│ LLM Generation (full)   ~2000ms                  │
│ ─────────────────────────────────────────────     │
│ Total                    ~2300ms ✅               │
└──────────────────────────────────────────────────┘
```

**Bottleneck**: LLM generation dominates. Optimize retrieval only after you've optimized the generation path (streaming, model selection, prompt length).

### 4.2 What to Keep Offline

| Component | Online / Offline | Rationale |
| :--- | :--- | :--- |
| Document parsing & chunking | Offline | CPU-intensive, not latency-sensitive |
| Embedding (documents) | Offline | Batch processing, can use larger models |
| Embedding (queries) | Online | Must be fast; use smaller/API models |
| Reranking | Online | Depends on query-document pairs |
| Index building (HNSW) | Offline | Expensive, done once per update |
| Metadata extraction | Offline | LLM-based, can use cheap models in batch |

## 5. Scaling Patterns

### 5.1 Read Scaling

```
                    ┌─────────────┐
User Queries ──────>│ Load Balancer│
                    └──────┬──────┘
                    ┌──────┼──────┐
                    ▼      ▼      ▼
               [Replica] [Replica] [Replica]
                    │      │      │
                    └──────┼──────┘
                           ▼
                    [Shared Index]
```

Most vector DBs support **read replicas**. Scale reads horizontally; writes go to a single primary.

### 5.2 Write Scaling (Ingestion)

```
New Docs ──> Message Queue (SQS/Kafka)
                    │
            ┌───────┼───────┐
            ▼       ▼       ▼
        [Worker] [Worker] [Worker]
            │       │       │
            └───────┼───────┘
                    ▼
            [Vector DB Upsert]
```

**Key**: Decouple ingestion from serving. Never let a bulk re-index impact query latency.

### 5.3 Multi-Tenancy

| Pattern | Isolation | Cost | Complexity |
| :--- | :--- | :--- | :--- |
| **Metadata filtering** (tenant_id field) | Low (shared index) | Lowest | Lowest |
| **Namespace/Collection per tenant** | Medium (logical separation) | Medium | Medium |
| **Database per tenant** | High (physical separation) | Highest | Highest |

**Default**: Start with metadata filtering. Move to namespaces only when you need strict data isolation (compliance) or tenants have vastly different data sizes.

## 6. Failure Mode Playbook

| Failure | Symptom | Root Cause | Architectural Fix |
| :--- | :--- | :--- | :--- |
| **Stale answers** | System ignores recently added docs | Index not refreshed; embedding pipeline delayed | Near-real-time ingestion pipeline with queue |
| **Wrong document, right answer** | Faithfulness low, relevancy high | LLM hallucinating from parametric knowledge | Stricter grounding prompt + faithfulness gate |
| **Right document, wrong answer** | Context precision high, relevancy low | Chunk too large; answer buried in noise | Smaller chunks + parent-child pattern |
| **Slow under load** | p95 latency >5s | Reranker bottleneck or DB not scaled | Cache frequent queries; add read replicas |
| **Contradictory answers** | Different answers for same question | Duplicate/outdated docs in index | Deduplication + version metadata + hard filtering |
| **Cross-tenant leakage** | Tenant A sees Tenant B's data | Missing or incorrect tenant_id filter | Enforce tenant filter at middleware level, not application code |

## 7. Production Checklist

Before going live, verify:

- [ ] **Golden Dataset** exists with 100+ Q&A pairs covering edge cases
- [ ] **RAGAS metrics** baselined: Faithfulness >0.85, Context Precision >0.75
- [ ] **Hybrid search** enabled (not vector-only) with reranking
- [ ] **Embedding model prefix** correctly applied (query vs. document)
- [ ] **Chunk size** validated against embedding model's max token limit
- [ ] **Metadata filtering** tested for tenant isolation and version control
- [ ] **Ingestion pipeline** decoupled from serving path
- [ ] **Monitoring** in place: latency p50/p95, token cost per query, retrieval hit rate
- [ ] **Fallback behavior** defined: what happens when retrieval returns 0 results?
- [ ] **Content freshness** SLA defined: how quickly must new docs be searchable?
