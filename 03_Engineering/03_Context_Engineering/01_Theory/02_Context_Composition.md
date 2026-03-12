# Context Composition

*Prerequisite: [01_Context_Window_Mechanics.md](01_Context_Window_Mechanics.md).*

---

Context composition is the discipline of deciding **what to put in the context window, in what order, and at what priority**. It is the core skill of context engineering.

## 1. The Anatomy of a Production Context

A production LLM call assembles context from multiple sources, each with a different role:

```
┌─────────────────────────────────────────────────────────────┐
│ LAYER 1: System Prompt          [Fixed, Cached]    ~500T    │
│   Role definition, persona, output format, constraints      │
├─────────────────────────────────────────────────────────────┤
│ LAYER 2: Background Knowledge   [Semi-static]      ~1000T   │
│   Domain rules, few-shot examples, reference data           │
├─────────────────────────────────────────────────────────────┤
│ LAYER 3: Retrieved Memory       [Dynamic]          ~2000T   │
│   Relevant facts from past sessions (vector DB lookup)      │
├─────────────────────────────────────────────────────────────┤
│ LAYER 4: RAG Context            [Dynamic]          ~4000T   │
│   Retrieved document chunks relevant to current query       │
├─────────────────────────────────────────────────────────────┤
│ LAYER 5: Conversation History   [Dynamic]          ~3000T   │
│   Recent turns (verbatim) + older turns (summarized)        │
├─────────────────────────────────────────────────────────────┤
│ LAYER 6: Tool Results           [Dynamic]          ~1000T   │
│   Outputs from function calls, API responses, code output   │
├─────────────────────────────────────────────────────────────┤
│ LAYER 7: Current User Query     [Dynamic]          ~200T    │
│   The immediate request                                     │
└─────────────────────────────────────────────────────────────┘
                                          TOTAL:    ~11700T
                                          Output Reserve: ~4300T
```

## 2. Ordering Principles

The order of layers matters because of attention patterns (see `01_Context_Window_Mechanics.md`).

### The Sandwich Pattern

Place the most critical instructions at **both ends** of the context:

```
[System Prompt + Core Instructions]   ← primacy bias: model reads carefully
[Background / Examples]
[Retrieved Context / Tool Results]
[Conversation History]
[Current Query]
[Reminder of Key Constraints]         ← recency bias: model acts on this
```

### Static Before Dynamic

Always place stable, cacheable content before dynamic content:

```
✅ [System][Few-shots][RAG][History][Query]
   ←── cacheable prefix ──→ ←── dynamic ──→

❌ [History][Query][System][Few-shots][RAG]
   ← changes every turn → ← cache miss →
```

### Recency for Relevance

The most relevant retrieved chunk should be placed **closest to the query**, not buried in the middle of a long RAG block.

## 3. Priority Hierarchy

When the total assembled context exceeds the budget, trim in this order:

| Priority | Layer | Trim Strategy |
| :--- | :--- | :--- |
| **P0 — Never trim** | System Prompt | Reduce the prompt itself if it doesn't fit |
| **P1 — Trim last** | Current User Query | Truncate only if extremely long (rare) |
| **P2 — Keep verbatim** | Recent History (last 2–4 turns) | Required for conversational coherence |
| **P3 — Reduce** | RAG Context | Fewer chunks (top-3 → top-1) or shorter excerpts |
| **P4 — Compress** | Retrieved Memory | Summarize or reduce top-k |
| **P5 — Trim first** | Old Conversation History | Summarize or drop entirely |
| **P6 — Conditional** | Tool Results | Keep only the most recent / most relevant |

## 4. Context Slots for Agent Systems

In agentic workflows, the context must also accommodate tool definitions and tool results. These compete for the same token budget:

```
┌──────────────────────────────────────────────────────────────┐
│ System Prompt + Agent Persona              ~500T             │
│ Tool Definitions (JSON schemas)            ~1000T  ← fixed   │
│ Scratchpad / Reasoning trace               ~2000T  ← grows   │
│ Tool Call Results (accumulated)            ~3000T  ← grows   │
│ Original Task + Constraints                ~500T             │
└──────────────────────────────────────────────────────────────┘
```

**Key challenge**: In long-running agents, tool results accumulate and can exhaust the context window. Strategies:
- **Result summarization**: Compress older tool outputs into a summary.
- **Selective retention**: Keep only the tool results that are still relevant to the current sub-task.
- **Context checkpointing**: Periodically summarize the entire scratchpad and restart with a clean context.

## 5. Multi-Modal Context Composition

When images, audio, or structured data are included, token budgets shift significantly:

| Content Type | Approximate Token Cost |
| :--- | :--- |
| Text (1 page) | ~500T |
| Image (low detail) | ~85T |
| Image (high detail, 1024×1024) | ~765T |
| Image (high detail, 2048×2048) | ~2000T |
| Audio (1 minute) | ~1500T (model-dependent) |

For multi-modal agents, image resolution management is a first-class context engineering concern.

---

## 6. Implementing Prefix Caching

Prefix caching requires placing static content first (see Section 2) and enabling it at the API level.

### Anthropic (`cache_control`)

```python
response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},  # Cache this block
        },
        {
            "type": "text",
            "text": FEW_SHOT_EXAMPLES,
            "cache_control": {"type": "ephemeral"},  # Cache this block too
        },
    ],
    messages=[
        {"role": "user", "content": user_query},  # Dynamic — not cached
    ],
)
# Check cache usage in response:
# response.usage.cache_creation_input_tokens  (first request)
# response.usage.cache_read_input_tokens      (subsequent requests)
```

**Rules**: Minimum 1024 tokens to be eligible for caching. Cache TTL is 5 minutes (ephemeral). Up to 4 cache breakpoints per request.

### OpenAI (automatic)

OpenAI caches automatically — no API changes needed. The first 1024+ tokens of the prompt are cached if they match a previous request exactly.

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # Cached automatically
        {"role": "user",   "content": FEW_SHOT_BLOCK},  # Cached automatically
        {"role": "user",   "content": user_query},      # Dynamic
    ],
)
# Check cache usage:
# response.usage.prompt_tokens_details.cached_tokens
```

**Rule**: The cached prefix must be byte-for-byte identical. Any change (even whitespace) invalidates the cache.

---

## Key References

1. **Anthropic. (2025). Effective Context Engineering for AI Agents.** Anthropic Engineering Blog.
2. **Liu, N. F., et al. (2024). Lost in the Middle.** *TACL, 12*, 157–173.
3. **OpenAI. (2024). Vision API — Token Costs for Images.** OpenAI Documentation.
