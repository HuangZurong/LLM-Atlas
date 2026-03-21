# Token Budget & Cost Management

*Prerequisite: [02_Context_Composition.md](02_Context_Composition.md).*
*Position in CE Pipeline: Step 2 (Budget & Sort) and Step 3 (Compress & Degrade)*

---

Token budget management is the practice of **allocating, monitoring, and optimizing** the finite token space across all context layers. It is where context engineering meets cost engineering.

## 1. The Budget Model

Every LLM call has a fixed total budget:

```
Total Budget = Model's Max Context Window (e.g., 128K tokens)

Allocation Example (16K budget):
┌──────────────────────────────────────────────┐
│ System Prompt (Fixed)              ~500T     │
│ Background / Few-shots (Fixed)     ~1000T    │
│ Retrieved Memory (Variable)        ~2000T    │
│ RAG Context (Variable)             ~4000T    │
│ Conversation History (Variable)    ~3000T    │
│ Current User Query                 ~200T     │
│ Tool Results (Variable)            ~1000T    │
│ ──────────────────────────────────────────── │
│ Reserved for Output                ~4300T    │
│ ──────────────────────────────────────────── │
│ TOTAL                              16000T    │
└──────────────────────────────────────────────┘
```

**Critical Rule**: Always reserve at least **20–30% of the budget** for output. If you fill 95% of the context with input, the model has no room to reason or generate a complete response.

## 2. Priority-Based Allocation Algorithm

```python
def allocate_budget(
    total_budget: int,
    system_tokens: int,
    query_tokens: int,
    output_reserve_ratio: float = 0.25,
) -> dict[str, int]:
    output_reserve = int(total_budget * output_reserve_ratio)
    available = total_budget - system_tokens - query_tokens - output_reserve

    return {
        "recent_history": int(available * 0.35),  # P2: keep verbatim
        "rag_context":    int(available * 0.35),  # P3: reduce chunks
        "memory":         int(available * 0.20),  # P4: compress
        "old_history":    int(available * 0.10),  # P5: summarize/drop
    }
```

When a layer exceeds its allocation, apply the trim strategy from the priority table in `02_Context_Composition.md`.

## 3. Compression Strategies

When trimming is needed, choose the right strategy per layer:

| Strategy | Mechanism | Information Loss | Cost |
| :--- | :--- | :--- | :--- |
| **Truncation** | Drop oldest tokens | High (loses beginning) | Zero |
| **Sliding Window** | Keep last N tokens | Medium (loses long-range) | Zero |
| **Extractive Summary** | Keep key sentences | Low-Medium | Low (regex/heuristic) |
| **Abstractive Summary** | LLM-generated summary | Low | Medium (LLM call) |
| **Entity Compression** | Extract structured facts | Very Low | Medium |
| **Semantic Deduplication** | Remove near-duplicate chunks | Low | Medium (embedding) |

**Rule of thumb**: Use cheap strategies (truncation, sliding window) for conversation history. Use higher-quality strategies (abstractive summary, entity compression) for long-term memory and critical documents.

## 4. Cost Optimization Strategies

### 4.1 Prefix Caching

Cache the static prefix (System Prompt + Few-shots) to avoid recomputing KV states on every request. See `01_Context_Window_Mechanics.md` for provider-specific setup.

**Expected savings**: 30–60% reduction in input token costs for high-traffic applications.

### 4.2 Tiered Model Strategy

Use cheaper models for context preprocessing, expensive models only for the final response:

```
User Query
    │
    ▼
[Cheap Model: gpt-4o-mini / claude-haiku]
    ├── Memory summarization
    ├── Entity extraction
    ├── Query classification
    └── RAG reranking
    │
    ▼
[Expensive Model: gpt-4o / claude-sonnet]
    └── Final user-facing response
```

### 4.3 Lazy Context Loading

Don't inject all context layers on every request. Load conditionally:

| Condition | Load Memory? | Load RAG? | Load Full History? |
| :--- | :--- | :--- | :--- |
| First turn of new session | ✅ (user profile) | Depends on query | ❌ |
| User references past context | ✅ | ❌ | ✅ |
| Factual / knowledge query | ❌ | ✅ | ❌ |
| Casual chitchat | ❌ | ❌ | ✅ (recent only) |

### 4.4 Semantic Caching

Cache full LLM responses for semantically similar queries. If a new query is within cosine distance < 0.05 of a cached query, return the cached response directly.

**Best for**: FAQ-style applications, repeated analytical queries on the same dataset.

## 5. Monitoring & Alerts

Track these metrics in production:

| Metric | Alert Threshold | Action |
| :--- | :--- | :--- |
| **Avg Input Tokens / Request** | >80% of context window | Review memory/RAG injection logic |
| **Output Truncation Rate** | >5% of responses | Increase output reserve ratio |
| **Context Assembly Latency** | >200ms p95 | Optimize retrieval or reduce top-k |
| **Memory Retrieval Latency** | >500ms p95 | Optimize vector index |
| **Cost per Conversation** | >$0.50 | Audit whether all injected context is necessary |
| **Cache Hit Rate** | <30% | Review prefix structure or semantic cache thresholds |

---

## Key References

1. **Anthropic. (2025). Prompt Caching.** Anthropic API Documentation.
2. **OpenAI. (2024). Prompt Caching.** OpenAI API Documentation.
3. **Zhu, Y., et al. (2023). Large Language Models Can Be Easily Distracted by Irrelevant Context.** *ICML 2023*.
