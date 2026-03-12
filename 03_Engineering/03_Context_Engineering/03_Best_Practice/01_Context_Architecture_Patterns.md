# Context Architecture Patterns

*Prerequisite: [../01_Theory/05_Dynamic_Context_Management.md](../01_Theory/05_Dynamic_Context_Management.md).*

---

## 1. The Four Context Architectures

### Pattern 1: Stateless (Fresh Assembly)
Rebuild context from scratch on every call. No history carried over.

**When to use**: Single-turn Q&A, document analysis, batch processing.
**Pros**: Simple, predictable, no state management overhead.
**Cons**: No conversational continuity.

### Pattern 2: Sliding Window
Keep the last N turns verbatim. Drop oldest turns when budget exceeded.

**When to use**: Short-to-medium conversations (< 20 turns), customer support.
**Pros**: Simple, always coherent recent context.
**Cons**: Loses long-range context abruptly.

### Pattern 3: Summary + Recent (Hybrid)
Compress old history into a running summary. Keep recent N turns verbatim.

**When to use**: Long conversations, coding assistants, research sessions.
**Pros**: Balances coherence and long-range context.
**Cons**: Summary is lossy; requires a compression step.

### Pattern 4: External Memory + RAG
Store all history externally (vector DB). Retrieve relevant turns per query.

**When to use**: Multi-session applications, enterprise assistants, personalization.
**Pros**: Unlimited history, highly relevant context injection.
**Cons**: Retrieval latency, may miss implicit context.

### Decision Tree (ASCII)

```
Is this a single-turn request?
├── YES → Pattern 1: Stateless
└── NO
    ├── Does the conversation span multiple sessions?
    │   ├── YES → Pattern 4: External Memory + RAG
    │   └── NO
    │       ├── Expected conversation length < 20 turns?
    │       │   ├── YES → Pattern 2: Sliding Window
    │       │   └── NO  → Pattern 3: Summary + Recent
```

---

## 2. The Static-First Rule

Always place static, cacheable content before dynamic content. This maximizes prefix cache hits.

```
✅ CORRECT (cache-friendly):
[System Prompt — static]
[Few-shot Examples — static]
[Retrieved Docs — changes per query]
[Conversation History — changes per turn]
[User Query — changes every call]
 ↑ long cacheable prefix ↑

❌ WRONG (cache-busting):
[User Query]
[Conversation History]
[System Prompt]
[Few-shot Examples]
 ↑ cache miss on every call ↑
```

**Impact**: With Anthropic's prompt caching (~90% discount on cached tokens), a 1000-token system prompt cached across 10,000 requests saves ~$1.35 at $0.015/1K tokens.

---

## 3. Context Hygiene Anti-Patterns

| Anti-Pattern | Problem | Fix |
| :--- | :--- | :--- |
| **Inject everything "just in case"** | Wastes tokens, dilutes attention, increases cost | Only inject what's relevant to the current query |
| **Fill context to 95%+ capacity** | No room for output; model truncates responses | Reserve 20–30% for output |
| **Instructions only in the middle** | Lost in the Middle — model ignores them | Use Sandwich Pattern: instructions at start AND end |
| **Skip NIAH testing** | Deploy long-context features that silently fail | Always run NIAH before deploying >32K context |
| **Same context structure for all tasks** | Over-provisioning for simple tasks, under for complex | Size context to task type (see `01_Theory/01_Context_Window_Mechanics.md`) |
| **Ignore tool result accumulation** | Agent loops exhaust context after 20–30 steps | Prune/summarize tool results proactively |
| **Inject raw tool output** | JSON blobs waste tokens | Extract only the relevant fields before injecting |
| **No compression fallback** | Hard failure when context overflows | Always have a graceful degradation path |

---

## 4. Production Checklist

Before deploying any feature that manages context:

- [ ] Context budget is explicitly defined (not relying on model defaults)
- [ ] Output reserve is at least 20% of total window
- [ ] Static content is placed before dynamic content (prefix caching enabled)
- [ ] Sandwich Pattern applied for critical instructions
- [ ] NIAH test run at target context length
- [ ] Compression strategy defined for each layer
- [ ] Compression triggers are set (soft/hard/emergency thresholds)
- [ ] Tool result accumulation handled in agent loops
- [ ] Context assembly latency measured (target: <100ms p95)
- [ ] Token usage logged per layer per request
- [ ] Alert configured for avg input tokens > 80% of window
- [ ] Graceful degradation tested (what happens when retrieval fails?)

---

## Key References

1. **Anthropic. (2025). Effective Context Engineering for AI Agents.** Anthropic Engineering Blog.
2. **Liu, N. F., et al. (2024). Lost in the Middle.** *TACL, 12*, 157–173.
