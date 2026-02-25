# Token Budget Management

*Prerequisite: [../01_Theory/02_Context_Window_Engineering.md](../01_Theory/02_Context_Window_Engineering.md).*

---

In production, the context window is a **shared resource** that must be carefully allocated across competing demands: system prompt, memory, RAG context, conversation history, and output space.

## 1. The Budget Model

Every LLM call has a fixed token budget:
```
Total Budget = Model's Max Context Window (e.g., 128K)

Allocation:
┌──────────────────────────────────────────────┐
│ System Prompt (Fixed)              ~500T     │
│ Memory Context (Variable)          ~2000T    │
│ RAG Retrieved Docs (Variable)      ~4000T   │
│ Conversation History (Variable)    ~3000T   │
│ Current User Query                 ~200T    │
│ ──────────────────────────────────────────── │
│ Reserved for Output                ~6300T   │
│ ──────────────────────────────────────────── │
│ TOTAL                              16000T   │
└──────────────────────────────────────────────┘
```

**Critical Rule**: Always reserve at least 20-30% of the budget for the model's output. If you fill 95% of the context with input, the model has no room to "think" or generate a complete response.

## 2. Priority-Based Allocation

When the total exceeds the budget, components must be **trimmed** in priority order:

| Priority | Component | Trim Strategy |
| :--- | :--- | :--- |
| **P0 (Never trim)** | System Prompt | Fixed. If it doesn't fit, reduce the prompt itself. |
| **P1 (Trim last)** | Current User Query | Truncate only if extremely long (rare). |
| **P2** | Recent Conversation (last 2-4 turns) | Keep verbatim for coherence. |
| **P3** | RAG Context | Reduce number of retrieved chunks (top-3 → top-1). |
| **P4** | Memory Context | Reduce retrieved memories or summarize. |
| **P5 (Trim first)** | Old Conversation History | Summarize or drop entirely. |

## 3. Dynamic Budget Allocation Algorithm

```python
def allocate_budget(total_budget: int, system_tokens: int, query_tokens: int):
    output_reserve = int(total_budget * 0.25)
    available = total_budget - system_tokens - query_tokens - output_reserve

    # Allocate remaining budget proportionally
    allocations = {
        "recent_history": int(available * 0.35),
        "rag_context":    int(available * 0.35),
        "memory":         int(available * 0.20),
        "old_history":    int(available * 0.10),
    }
    return allocations
```

## 4. Cost Optimization Strategies

### 4.1 Tiered Model Strategy
- Use a **cheap model** (GPT-4o-mini) for memory summarization and entity extraction.
- Use the **expensive model** (GPT-4o) only for the final user-facing response.

### 4.2 Aggressive Caching
- Cache the System Prompt + Few-shot examples (Prefix Caching).
- Cache frequently retrieved RAG chunks (Semantic Cache from `01_LLMs`).

### 4.3 Lazy Memory Loading
Don't inject memory into every request. Only load memory when:
1. The user explicitly references past context ("What did we decide?").
2. The query is ambiguous and needs historical context to disambiguate.
3. The conversation is in its first turn of a new session (load user profile).

## 5. Monitoring & Alerts

Track these metrics in production:

| Metric | Alert Threshold | Action |
| :--- | :--- | :--- |
| **Avg Input Tokens** | >80% of context window | Review memory/RAG injection logic |
| **Output Truncation Rate** | >5% of responses | Increase output reserve |
| **Memory Retrieval Latency** | >500ms p95 | Optimize vector index or reduce top-k |
| **Cost per Conversation** | >$0.50 | Review if all injected context is necessary |
