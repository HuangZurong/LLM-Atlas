# Dynamic Context Management

*Prerequisite: [03_Token_Budget_and_Cost.md](03_Token_Budget_and_Cost.md) | [04_Long_Context_Techniques.md](04_Long_Context_Techniques.md).*
*Position in CE Pipeline: Step 3 (Compress & Degrade) and Step 4 (Assemble & Observe)*

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

## 6. Schema-Driven State Tracking

The evolution from "text accumulation" to "structured state management" represents a paradigm shift in context engineering. Instead of appending conversation history verbatim, maintain a **fixed-size state machine** that evolves through interactions.

### 6.1 The Paradigm Shift

| Traditional Approach | Schema-Driven Approach |
| :--- | :--- |
| Append history to prompt | Track structured state object |
| O(N) context growth | O(1) context complexity |
| Full transcripts | Condensed facts |
| Lost in the Middle risk | High-signal, low-noise |

**Core insight**: Natural language is an inefficient storage format for state. A JSON object of extracted facts is 10–100× more token-efficient than the equivalent conversation transcript.

### 6.2 Hybrid Schema Architecture

Production systems use a **hybrid schema** that balances code stability with LLM flexibility:

```
┌─────────────────────────────────────────────────────────────┐
│                    User Session Schema                       │
├─────────────────────────────────────────────────────────────┤
│  core_state: CoreState          ← Strong-typed, code-owned  │
│    ├── intent: str              ← Business logic depends on │
│    ├── destination: str         ← Used for API calls        │
│    ├── budget: float | None     ← Must match schema exactly │
│    └── status: enum             ← State machine control     │
├─────────────────────────────────────────────────────────────┤
│  dynamic_profile: dict          ← Weak-typed, LLM-owned     │
│    ├── dietary_preference: str  ← Discovered at runtime     │
│    ├── allergies: list[str]     ← User volunteered info     │
│    └── pet_info: dict           ← Nested, ad-hoc structure  │
├─────────────────────────────────────────────────────────────┤
│  recent_context: str            ← Emotional/mood summary    │
└─────────────────────────────────────────────────────────────┘
```

**Design principles**:
- **Core State**: Fixed schema defined by engineering. Maps directly to API calls, database queries, and business logic. The LLM can only *modify* existing fields, not add new ones.
- **Dynamic Profile**: Free-form dictionary the LLM can extend. Records user preferences, constraints, and context discovered through conversation.
- **Recent Context**: Brief summary of the last 2–3 turns for emotional continuity (e.g., "User changed destination from Japan to Thailand, seems excited").

### 6.3 Tiered Model Routing

Schema updates are a natural fit for **smaller, cheaper models**:

```
User Query
    │
    ├──► [Small Model Pipeline] (GPT-4o-mini, Claude Haiku)
    │    Input: [Current Schema] + [User Query]
    │    Output: Updated Schema (JSON diff)
    │    Cost: ~$0.0001 per update
    │
    ├──► [Code Layer]
    │    Parse core_state → Trigger APIs if fields complete
    │    Validate dynamic_profile → Merge, dedupe
    │
    └──► [Large Model Pipeline] (GPT-4o, Claude Sonnet)
         Input: [Minimal Schema] + [API Results] + [Query]
         Output: Final response
         Cost: ~$0.01 per response
```

**The schema acts as a compression layer**: The large model never sees raw conversation history—only the distilled state.

### 6.4 Handling Key Proliferation

When the LLM freely adds keys to `dynamic_profile`, you eventually get semantic duplicates:

```json
// Problem: Same concept, different keys
{
  "food_taboos": ["seafood"],
  "cannot_eat": ["shellfish"],
  "dietary_restrictions": ["no shrimp"]
}
```

**Solution: Memory Consolidation**

Run a background consolidation pipeline (triggered by CRON or every N turns):

```python
CONSOLIDATION_PROMPT = """
You are a memory consolidation engine. Given a dynamic profile with potentially redundant keys:

1. Identify keys with overlapping semantics
2. Merge them into a single canonical key
3. Preserve all values (union)
4. Output a cleaned, deduplicated profile

Original: {dynamic_profile}
Consolidated:
"""
```

**Solution: Tree-Structured Memory**

Instead of a flat dictionary, enforce a hierarchical structure with predefined parent nodes:

```python
DYNAMIC_PROFILE_SCHEMA = {
    "dietary_profile": {},      # All food-related preferences
    "travel_preferences": {},   # All travel-related preferences
    "personal_info": {},        # All personal details
}
# LLM can add keys, but only under these parent nodes
```

This mimics cognitive schemas: "pollen allergy" and "seafood allergy" are both filed under `dietary_profile.allergies`.

### 6.5 Implementation Pattern

```python
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from enum import Enum

class SessionStatus(str, Enum):
    EXPLORING = "exploring"
    COLLECTING = "collecting"
    CONFIRMED = "confirmed"
    COMPLETED = "completed"

class CoreState(BaseModel):
    """Strong-typed state for business logic. LLM cannot add fields."""
    intent: Optional[str] = Field(default=None, description="User's primary goal")
    destination: Optional[str] = Field(default=None, description="Target location")
    departure_date: Optional[str] = Field(default=None)
    budget: Optional[float] = Field(default=None)
    status: SessionStatus = Field(default=SessionStatus.EXPLORING)

class SessionSchema(BaseModel):
    """Full session state with hybrid typing."""
    session_id: str
    core_state: CoreState                           # Code-owned
    dynamic_profile: Dict[str, Any] = Field(default_factory=dict)  # LLM-owned
    recent_summary: str = ""

    def is_ready_for_api(self) -> bool:
        """Check if core state has enough info for external API calls."""
        return all([
            self.core_state.destination,
            self.core_state.departure_date,
            self.core_state.budget is not None,
        ])

# State update prompt for small model
UPDATE_STATE_PROMPT = """
You are a state management engine. Update the session schema based on the user's latest message.

Rules:
1. core_state: Only modify existing fields. Never add new keys.
2. dynamic_profile: You may add new keys to record user preferences (e.g., allergies, preferences).
3. If you detect duplicate semantics in dynamic_profile, merge them.
4. Update recent_summary with a 1-sentence emotional/contextual note.

Current Schema: {current_schema}
User Message: {user_message}

Output the updated schema as JSON:
"""
```

### 6.6 Research Frontiers

| Research Direction | Key Paper / Project | Core Idea |
| :--- | :--- | :--- |
| **Self-Evolving Memory** | Evo-Memory (UIUC + Google DeepMind, 2025) | Test-time learning: agent runs `ReMem (Action-Think-Memory Refine)` pipeline during idle time |
| **MemSkill** | MemSkill (2026) | Memory as learnable skill; agent maintains its own skill bank for schema evolution |
| **MemTree** | MemTree (2024) | Dynamic tree structure mimics cognitive schemas; auto-groups related concepts under parent nodes |
| **Ontology-Driven Memory** | Knowledge Graph Memory | Fixed ontology skeleton + dynamic instance nodes; best for multi-agent coordination |
| **LangMem** | LangChain LangMem | Production framework with Profiles (typed) + Collections (untyped) |

**Key trend**: Memory is becoming *active*—not just storage, but a reasoning substrate the agent can query, reorganize, and refine.

## 7. Context Isolation in Multi-Agent Systems

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

## 8. Context Observability

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
4. **Evo-Memory. (2025). Self-Evolving Memory for LLM Agents.** UIUC + Google DeepMind.
5. **MemTree. (2024). Dynamic Tree-Structured Memory for Conversational Agents.**
6. **LangChain. (2025). LangMem: Semantic Memory for AI Agents.** LangChain Documentation.
