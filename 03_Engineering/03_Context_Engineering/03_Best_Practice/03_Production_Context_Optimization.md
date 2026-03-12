# Production Context Optimization

*Prerequisite: [02_Context_Quality_and_Evaluation.md](02_Context_Quality_and_Evaluation.md).*

---

## 1. Latency Optimization

Context assembly involves multiple I/O operations (vector DB lookup, memory retrieval, history fetch). These must be parallelized.

### Async Parallel Assembly

```python
import asyncio

async def assemble_context_async(query: str, session_id: str) -> dict:
    # Run all retrievals in parallel
    memory_task = asyncio.create_task(retrieve_memory(session_id, query))
    rag_task = asyncio.create_task(retrieve_rag(query))
    history_task = asyncio.create_task(fetch_history(session_id))

    memory, rag_results, history = await asyncio.gather(
        memory_task, rag_task, history_task
    )
    return {"memory": memory, "rag": rag_results, "history": history}
```

**Target latency**: Context assembly < 100ms p95. If exceeded, profile which retrieval step is the bottleneck.

### Pre-warming the Prefix Cache

For high-traffic applications, send a "warm-up" request with just the static prefix (system prompt + few-shots) at startup. This ensures the KV cache is populated before real traffic arrives.

### Lazy Loading

Don't load all context layers on every request:

| Condition | Skip |
| :--- | :--- |
| First turn, simple query | Memory retrieval |
| Factual Q&A | Conversation history |
| Casual chitchat | RAG retrieval |
| Short, unambiguous query | Memory retrieval |

---

## 2. Cost Optimization

### Prefix Caching ROI

```
Daily savings = (cached_tokens_per_request × requests_per_day × cache_discount × price_per_token)

Example (Anthropic claude-3-5-sonnet):
  cached_tokens = 1000 (system prompt + few-shots)
  requests/day  = 10,000
  cache_discount = 90% (Anthropic)
  price          = $0.003 / 1K input tokens

  Daily savings = 1000 × 10,000 × 0.90 × $0.000003 = $27/day = ~$810/month
```

### Tiered Model Strategy

| Task | Model | Rationale |
| :--- | :--- | :--- |
| Memory summarization | claude-haiku / gpt-4o-mini | Cheap, fast, good enough |
| Entity extraction | claude-haiku / gpt-4o-mini | Structured output, low complexity |
| RAG reranking | claude-haiku / gpt-4o-mini | Binary relevance judgment |
| Query classification | claude-haiku / gpt-4o-mini | Simple classification |
| Final user response | claude-sonnet / gpt-4o | Quality matters here |

### Semantic Deduplication

Before injecting multiple RAG chunks, remove near-duplicates:

```python
def deduplicate_chunks(chunks: list[str], threshold: float = 0.92) -> list[str]:
    """Remove chunks with cosine similarity > threshold to any already-selected chunk."""
    embeddings = embed(chunks)
    selected = [0]
    for i in range(1, len(chunks)):
        similarities = cosine_similarity(embeddings[i], [embeddings[j] for j in selected])
        if max(similarities) < threshold:
            selected.append(i)
    return [chunks[i] for i in selected]
```

---

## 3. Reliability Patterns

### Graceful Degradation

Define fallback behavior for each context layer:

| Layer | Primary | Fallback |
| :--- | :--- | :--- |
| RAG context | Vector DB retrieval | Empty (model uses parametric knowledge) |
| Memory | Memory service | Empty (stateless response) |
| History | Session store | Last 2 turns from request payload |
| System prompt | Config service | Hardcoded default |

### Context Validation

Before sending to the LLM, validate the assembled context:

```python
def validate_context(context: str) -> list[str]:
    issues = []
    tokens = count_tokens(context)
    if tokens > MAX_INPUT_TOKENS * 0.95:
        issues.append(f"Context too large: {tokens} tokens")
    if detect_prompt_injection(context):
        issues.append("Potential prompt injection detected")
    if detect_pii(context):
        issues.append("PII detected in context")
    return issues
```

### Idempotent Assembly

Context assembly should be deterministic given the same inputs. Avoid:
- Random sampling in retrieval (use deterministic top-k)
- Time-dependent content in static layers
- Non-deterministic compression (use temperature=0 for summarization)

---

## 4. Observability

### Key Metrics

| Metric | Alert Threshold | Dashboard |
| :--- | :--- | :--- |
| Avg input tokens / request | >80% of window | Yes |
| Output truncation rate | >5% | Yes |
| Context assembly latency p95 | >200ms | Yes |
| Cache hit rate | <30% | Yes |
| Context relevance score avg | <0.5 | Yes |
| Cost per 1K requests | >budget | Yes |
| Compression trigger rate | >20% of requests | Warning |

### Distributed Tracing

Instrument each assembly step as a span:

```
request_id: abc123
├── retrieve_memory        12ms
├── retrieve_rag           45ms  ← bottleneck
├── fetch_history           8ms
├── assemble_context        3ms
│   ├── token_count        1ms
│   ├── priority_trim      1ms
│   └── sandwich_wrap      1ms
└── llm_call             820ms
    ├── ttft              340ms
    └── generation        480ms
```

### Context Diff Logging

Log what changed in the context between turns to debug unexpected behavior:

```python
context_diff = {
    "turn": turn_number,
    "added_layers": [...],
    "removed_layers": [...],
    "token_delta": current_tokens - previous_tokens,
    "compression_applied": bool,
}
```

### Cost Attribution

Break down cost per context layer to identify optimization opportunities:

```
Request cost breakdown:
  system_prompt:    $0.0015  (500T, cached → $0.00015 effective)
  rag_context:      $0.0120  (4000T, not cached)
  history:          $0.0090  (3000T, not cached)
  user_query:       $0.0006  (200T)
  output:           $0.0240  (800T at output rate)
  ─────────────────────────
  Total:            $0.0471
```

---

## Key References

1. **Anthropic. (2025). Prompt Caching.** Anthropic API Documentation.
2. **OpenAI. (2024). Latency Optimization Guide.** OpenAI Documentation.
3. **Arize AI. (2024). LLM Observability Best Practices.** Arize Documentation.
