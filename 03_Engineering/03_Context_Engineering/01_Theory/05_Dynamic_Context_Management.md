# Dynamic Context Management

*Prerequisite: [03_Token_Budget_and_Cost.md](03_Token_Budget_and_Cost.md) | [04_Long_Context_Techniques.md](04_Long_Context_Techniques.md).*

---

Dynamic context management is the runtime orchestration layer that assembles, updates, and maintains the context window across the lifecycle of a conversation or agent session.

## 1. Context as a State Machine

Think of the context window as a **managed state** that evolves over time, not a static prompt:

```
Session Start
     │
     ▼
[Initialize]  Load system prompt, user profile, session config
     │
     ▼
[Assemble]    For each turn: retrieve memory + RAG + format history
     │
     ▼
[Execute]     LLM call with assembled context
     │
     ▼
[Update]      Store new turn, extract entities, update memory
     │
     ▼
[Compress?]   If context > threshold: summarize / prune
     │
     └──────────────────────────────────────────────────────→ Next Turn
```

## 2. Turn-by-Turn Context Evolution

### Single-Turn (Stateless)

Each request is independent. Context is assembled fresh from scratch:

```python
context = build_context(
    system_prompt=SYSTEM_PROMPT,
    rag_results=retrieve(query),
    user_query=query,
)
response = llm.call(context)
```

### Multi-Turn (Stateful)

Context accumulates across turns. The key challenge is managing growth:

```
Turn 1:  [System][Query1][Response1]                          ~1K
Turn 2:  [System][Query1][Response1][Query2][Response2]       ~2K
Turn N:  [System][Q1][R1]...[QN-1][RN-1][QN]                 ~NK
```

At some point, the accumulated history exceeds the budget. The context manager must intervene.

## 3. Context Compression Triggers

Define explicit thresholds that trigger compression:

```python
COMPRESSION_THRESHOLDS = {
    "soft_limit": 0.70,   # Start compressing old history
    "hard_limit": 0.85,   # Aggressive compression: summarize everything
    "emergency":  0.95,   # Drop non-essential content immediately
}

def should_compress(current_tokens: int, max_tokens: int) -> str:
    ratio = current_tokens / max_tokens
    if ratio >= COMPRESSION_THRESHOLDS["emergency"]:
        return "emergency"
    elif ratio >= COMPRESSION_THRESHOLDS["hard_limit"]:
        return "hard"
    elif ratio >= COMPRESSION_THRESHOLDS["soft_limit"]:
        return "soft"
    return "none"
```

## 4. Context Window Strategies by Session Type

| Session Type | History Strategy | Memory Strategy | Typical Budget |
| :--- | :--- | :--- | :--- |
| **Single Q&A** | None | None | 4–8K |
| **Short chat** | Full verbatim | None | 16–32K |
| **Long conversation** | Recent verbatim + old summarized | Entity extraction | 32–64K |
| **Document analysis** | Minimal | None | 32–128K |
| **Agent task** | Scratchpad + checkpoints | Task state | 16–64K |
| **Multi-session** | Summary only | Vector DB | 8–16K per session |

## 5. Agent Context Management

Agents have unique context challenges because they execute multi-step plans with tool calls.

### The Scratchpad Pattern

Maintain a structured scratchpad that separates reasoning from results:

```
[System Prompt]
[Task Definition]
[Tool Schemas]
─────────────────────────────────────────
SCRATCHPAD:
Thought: I need to find the user's order history first.
Action: search_orders(user_id="u123")
Result: [Order #1001, #1002, #1003]

Thought: Now I need to check the status of #1001.
Action: get_order_status(order_id="1001")
Result: {"status": "shipped", "eta": "2025-03-15"}
─────────────────────────────────────────
[Current Step]
```

### Scratchpad Summarization

When the scratchpad grows too large, compress completed reasoning chains:

```
BEFORE (verbose):
  Thought: I need X. Action: tool_a(). Result: {...full JSON...}
  Thought: Based on X, I need Y. Action: tool_b(). Result: {...full JSON...}
  Thought: X and Y together mean Z. Action: tool_c(). Result: {...full JSON...}

AFTER (compressed):
  COMPLETED: Retrieved X via tool_a, Y via tool_b, Z via tool_c.
  KEY FACTS: [fact1, fact2, fact3]
  CURRENT STATE: Ready to synthesize final answer.
```

### Context Checkpointing

For very long agent runs (>30 steps), periodically checkpoint:

1. Summarize all completed steps into a structured state object.
2. Save the state to external storage (Redis, DB).
3. Reset the context window.
4. Inject only the checkpoint summary + current task.

```python
checkpoint = {
    "task": original_task,
    "completed_steps": summarize_scratchpad(scratchpad),
    "key_findings": extract_key_facts(tool_results),
    "current_subtask": current_step,
    "remaining_steps": plan[current_index:],
}
```

## 6. Context Isolation in Multi-Agent Systems

When multiple agents share information, context boundaries must be explicit:

```
Orchestrator Context:
  [System: Orchestrator role]
  [Task decomposition]
  [Sub-agent results: SUMMARY ONLY]   ← not full sub-agent contexts

Sub-agent Context:
  [System: Specialist role]
  [Assigned sub-task]
  [Relevant tools + data]
  [No access to other sub-agents' contexts]
```

**Key principle**: Sub-agent contexts are isolated. The orchestrator receives only structured summaries, not raw sub-agent transcripts. This prevents context pollution and keeps each agent's window focused.

## 7. Context Observability

Instrument context management for debugging and optimization:

```python
# Log context composition for every LLM call
context_log = {
    "timestamp": ...,
    "session_id": ...,
    "total_tokens": ...,
    "layer_breakdown": {
        "system": 500,
        "memory": 1800,
        "rag": 3200,
        "history": 2900,
        "query": 180,
        "output_reserve": 4420,
    },
    "compression_applied": "soft",
    "cache_hit": True,
}
```

Track these over time to identify context bloat, inefficient retrieval, and compression opportunities.

---

## Key References

1. **Anthropic. (2025). Effective Context Engineering for AI Agents.** Anthropic Engineering Blog.
2. **Yao, S., et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models.** *ICLR 2023*.
3. **Wang, L., et al. (2024). A Survey on Large Language Model based Autonomous Agents.** *Frontiers of Computer Science*.
