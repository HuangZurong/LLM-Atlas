# 10 · CE Framework Design Document

## 1. Executive Summary

The **Context Engineering (CE) Framework** is a configurable, pluggable, and vendor-agnostic lifecycle manager for LLM agent context. While Prompt Engineering focuses on "how" to talk to a model, this framework focuses on "what" information, in what "order", and under what "constraints" that information enters the limited attention budget of the LLM.

The framework aims to solve the problem of **Context Rot** and **Inefficient Budgeting** by providing a systematic way to load, select, compress, and isolate context across multiple agents and long-running sessions.

---

## 2. Design Principles

- **"Do the simplest thing that works"**: Inspired by Anthropic. The framework is opt-in; simple agents should incur zero overhead.
- **Configuration-Driven Mechanical Operations**: Budgeting, caching, and truncation rules are defined in YAML.
- **Plugin-Driven Semantic Operations**: Summarization, reranking, and distillation are implemented as pluggable strategies.
- **Vendor-Agnostic**: Works with any LLM provider (Gemini, Claude, GPT, etc.) and any orchestration framework (ADK, LangGraph, OpenAI SDK).
- **Observable by Default**: Integrated token counting and precision metrics for every context assembly step.
- **Progressive Degradation**: Context is managed through a "chain of degradation" (Clear -> Summarize -> Drop) rather than all-or-nothing truncation.

---

## 3. Architecture Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                    Context Engineering Framework             │
│                                                              │
│  ┌───────────────────┐      ┌─────────────────────────┐      │
│  │  Context Registry │      │     Budget Manager      │      │
│  │ (Sources & Cache) │      │  (Counting & Triggers)  │      │
│  └─────────┬─────────┘      └────────────┬────────────┘      │
│            │                             │                   │
│  ┌─────────▼─────────┐      ┌────────────▼────────────┐      │
│  │  Retrieval Engine │      │   Compression Pipeline  │      │
│  │ (Recall & Rerank) │      │   (Degradation Chain)   │      │
│  └─────────┬─────────┘      └────────────┬────────────┘      │
│            │                             │                   │
│  ┌─────────▼─────────┐      ┌────────────▼────────────┐      │
│  │   State Router    │      │     Memory Manager      │      │
│  │ (Scopes & Merging)│      │  (Hierarchical Tiers)   │      │
│  └─────────┬─────────┘      └────────────┬────────────┘      │
│            │                             │                   │
│            └──────────────┬──────────────┘                   │
│                           │                                  │
│                ┌──────────▼──────────┐                       │
│                │ Isolation Controller│                       │
│                │  (Context Sandboxing)                       │
│                └──────────┬──────────┘                       │
└───────────────────────────┼──────────────────────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │     Integration Layer     │
              │ (ADK / LangGraph / OpenAI)│
              └───────────────────────────┘
```

### 3.1 The Seven Core Modules

1.  **Context Registry**: Manages source registration (DB, File, History) with a tiered priority system and TTL-based caching.
2.  **Budget Manager**: Tracks token usage against per-agent and per-source limits. Triggers compression when thresholds are crossed.
3.  **Retrieval Engine**: Implements a two-stage pipeline (Semantic Recall -> Cross-Encoder Rerank) to select high-signal chunks.
4.  **Compression Pipeline**: Executes a progressive sequence of strategies to fit information into the budget.
5.  **Memory Manager**: Manages information across three tiers: Working (Window), Short-term (Session), and Persistent (Long-term).
6.  **Isolation Controller**: Controls context visibility between agents (None, Partial, Full) to prevent cross-contamination.
7.  **State Router**: Orchestrates structured data flow using scoped prefixes (`app:`, `user:`, `temp:`) and custom merge strategies.

---

## 4. Module Specifications

### 4.1 Context Registry
The Registry defines **where** information comes from and its **inherent value**.

- **Tiered Priority**: `CRITICAL` (never dropped) down to `LOW` (first to be dropped).
- **Just-in-Time Loading**: Inspired by Anthropic; only loads tool definitions or documents when explicitly referenced.
- **Cache with TTL**: Memory or Redis-backed cache for repeated source loading.

### 4.2 Budget Manager
The Budget Manager is the **Operating System's Memory Controller**.

- **Threshold Triggers**: `Warning` (60%), `Compress` (75%), `Compaction` (90%).
- **Pluggable Tokenizers**: Supports Tiktoken, Google GenAI, and HuggingFace tokenizers.

### 4.3 Retrieval Engine
Inspired by Cursor's two-stage pipeline for precision.

- **Stage 1 (Recall)**: Fast vector search retrieves candidate chunks.
- **Stage 2 (Rerank)**: A Cross-Encoder or fast LLM re-scores candidates to pick Top-K high-signal chunks.
- **AST-Aware**: Optional structural chunking for code contexts.

### 4.4 Compression Pipeline
Implements a **Progressive Degradation Chain**.

```yaml
compression:
  chain:
    - strategy: clear_thinking      # remove old reasoning blocks
    - strategy: clear_tool_results  # keep only latest N results
      keep: 3
    - strategy: degrade_images      # High-res -> Low-res -> Text captions (for multimodal)
    - strategy: perplexity_pruning  # info-theoretic compression of low-signal tokens
    - strategy: summarize           # LLM-based abstractive summary of history
    - strategy: drop_optional       # discard Priority.LOW sources
```

### 4.5 Memory Manager
A four-phase lifecycle inspired by OpenAI: **Inject -> Distill -> Consolidate -> Reinject**.

- **Distillation**: Periodically extracts "facts" from the session history.
- **Consolidation**: Moving session-level facts to user-level persistent storage.

### 4.6 Isolation Controller
Controls the "wall" between agents.

- **Level 1 (None)**: Shared conversation history.
- **Level 2 (Partial)**: Filtered history (e.g., stripping tool call JSON).
- **Level 3 (Full)**: No history, only injected state (Google pattern).

### 4.7 State Router
Manages structured context bus.

- **Scopes**: `app:` (global), `user:` (persistent), `session:` (this run), `temp:` (sub-task).
- **Merge Strategies**: `overwrite`, `append` (list), `reducer` (custom logic).

---

## 5. Configuration Schema (Example)

```yaml
# context-engineering.yaml
global:
  default_budget: 8000
  compaction_threshold: 0.75

sources:
  user_profile:
    type: state
    keys: ["user:name", "user:history"]
    priority: critical

  code_snippets:
    type: retrieval
    engine: "two_stage"
    params: { top_k: 5 }
    priority: high

agents:
  coder:
    budget: 16000
    isolation: partial
    sources: [user_profile, code_snippets]
    overflow:
      - strategy: clear_thinking
      - strategy: clear_tool_results
      - strategy: perplexity_pruning
      - strategy: summarize
```

---

## 6. Integration Patterns

- **Google ADK**: via `before_agent_callback`.
- **OpenAI Agents SDK**: via `RunContextWrapper`.
- **LangGraph**: via state reducers and node wrappers.

---

## 7. Implementation Roadmap

1.  **Phase 1 (Core)**: Registry, Budgeting, and Truncation.
2.  **Phase 2 (Selection)**: Embedding search and Cross-Encoder reranking.
3.  **Phase 3 (Lifecycle)**: 4-phase memory pipeline and State Router.
4.  **Phase 4 (Advanced)**: Implement academic paradigms (Perplexity pruning, structured data verbalization, and multi-modal budget degradation).
5.  **Phase 5 (Autonomous)**: Automated context optimization (DSPy style self-healing context).

---

## 8. References

- [04_Anthropic_CE_Practices](04_Anthropic_CE_Practices.md)
- [05_OpenAI_CE_Practices](05_OpenAI_CE_Practices.md)
- [06_Google_CE_Practices](06_Google_CE_Practices.md)
- [07_Cursor_CE_Practices](07_Cursor_CE_Practices.md)
- [08_Frameworks_CE_Practices](08_Frameworks_CE_Practices.md)
- [09_CE_Cross_Comparison](09_CE_Cross_Comparison.md)
