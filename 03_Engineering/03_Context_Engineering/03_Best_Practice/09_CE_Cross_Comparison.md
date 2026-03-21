# 09 · Context Engineering Cross-Comparison

*A structured comparison of CE practices across Anthropic, OpenAI, Google/ADK, Cursor, and major agent frameworks.*

---

## 1. Philosophy Comparison

| Vendor | Core Metaphor | CE Philosophy |
|--------|--------------|---------------|
| **Anthropic** | Context is a finite attention budget | Smallest high-signal token set; Context Rot is the core enemy |
| **OpenAI** | LLM is CPU, Context is RAM, developer is OS | Write / Select / Compress / Isolate — four operations |
| **Google/ADK** | State is the Context Bus | Structured state passing > conversation history; three-scope prefixes |
| **Cursor** | Tiered retrieval pipeline | Better selection beats larger windows; two-stage retrieval |
| **LangGraph** | Graph State with reducers | Per-field merge strategies control context flow |
| **CrewAI** | Four-layer memory architecture | Short-term / Long-term / Entity / User memory separation |
| **MemGPT/Letta** | Virtual memory management | Agent self-manages page in/out of context |
| **Devin** | Environment as memory | Filesystem/terminal/browser are external storage; re-read on demand |
| **DSPy** | Programmatic optimization | Auto-search for optimal context configuration |

---

## 2. Context Composition Strategies

### 2.1 What Goes Into Context

| Component | Anthropic | OpenAI | Google/ADK | Cursor |
|-----------|-----------|--------|------------|--------|
| **System prompt** | "Right altitude" — contract format | CTCO pattern (Context/Task/Constraints/Output) | Instruction with `{state}` template variables | .cursorrules project-level rules |
| **Tools** | Minimal set, excellent descriptions, < 25K token responses | Tool search for lazy loading; fine-tune to reduce tokens | Tools defined on Agent; cached via ContextCacheConfig | N/A (tools are IDE actions) |
| **Few-shot examples** | Diverse and canonical, not exhaustive | Included in system prompt | Via instruction text | N/A |
| **History** | Compaction + context editing | Trimming or compression | `ContextFilterPlugin`, `include_contents="none"` | Token budget allocation per source |
| **Retrieved data** | Just-in-Time via tools | RAG via tool outputs | Memory services (RAG, Memory Bank) | Two-stage retrieval (embedding + reranker) |
| **External memory** | `claude-progress.txt`, CLAUDE.md | YAML frontmatter + Markdown notes | `app:`/`user:`/`temp:` state scopes | .cursorrules |

### 2.2 Dynamic vs. Static Context

| Approach | Who Uses It | How |
|----------|-------------|-----|
| **Static instructions** | All | System prompt, rules files |
| **Dynamic instructions** | OpenAI, Google/ADK | Functions that generate instructions at runtime |
| **Template injection** | Google/ADK | `{variable}` in instruction, resolved from state |
| **Tool-based loading** | Anthropic, OpenAI, Devin | Agent uses tools to fetch context on demand |
| **Retrieval pipeline** | Cursor, CrewAI | Embedding search → reranking → injection |

---

## 3. Context Window Management

### 3.1 Compression / Compaction

| Vendor | Mechanism | Trigger | What's Preserved | What's Discarded |
|--------|-----------|---------|-----------------|-----------------|
| **Anthropic** | LLM summarization | Approaching window limit | Architecture decisions, unresolved bugs, implementation details | Redundant tool outputs, old messages |
| **OpenAI** | Encrypted compaction item | `compact_threshold` (configurable token count) | User messages verbatim; latent model understanding in encrypted item | Assistant messages, tool calls, reasoning |
| **Google/ADK** | `ContextFilterPlugin` | `num_invocations_to_keep` | Last N invocations | Older invocations |
| **LangGraph** | Summarization node | Custom trigger | Summary of older messages | Original older messages |
| **AutoGen** | Head-and-Tail | Buffer size | First K + Last N messages | Middle messages |

### 3.2 Degradation Chains

**Anthropic** (most sophisticated):
```
Level 1: Clear old tool results (replace with placeholder)
Level 2: Clear old thinking blocks
Level 3: LLM summarization of history
Level 4: Drop optional context sources
Level 5: Full compaction with progress file
```

**OpenAI**:
```
Level 1: Context trimming (drop old turns)
Level 2: Context compression (summarize old turns)
Level 3: Server-side compaction (encrypted compaction item)
```

**Google/ADK**:
```
Level 1: ContextFilterPlugin (keep last N invocations)
Level 2: include_contents="none" (strip all history, inject via state)
Level 3: Branch-based isolation (filter by agent branch)
```

---

## 4. Multi-Agent Context Isolation

| Vendor | Isolation Mechanism | Granularity |
|--------|-------------------|-------------|
| **Anthropic** | Separate context windows per subagent; orchestrator-worker pattern | Full isolation (separate sessions) |
| **OpenAI** | `input_filter` on handoffs; `remove_all_tools`; nested handoff history | Configurable per handoff |
| **Google/ADK** | `include_contents="none"` + state injection; branch-based event filtering | Per-agent or per-branch |
| **LangGraph** | Graph state scoping; each node sees only its input state | Per-node |
| **CrewAI** | Task-scoped context; each agent gets task description + relevant memories | Per-task |
| **AutoGen** | `ChatCompletionContext` protocol; custom context classes | Per-agent |
| **MemGPT** | Each agent manages its own memory independently | Per-agent (self-managed) |

---

## 5. Memory Architecture

### 5.1 Memory Tiers

| Tier | Anthropic | OpenAI | Google/ADK | CrewAI | MemGPT |
|------|-----------|--------|------------|--------|--------|
| **Working** (in context window) | Current context | Current messages | Current state + history | Current task context | Main context (core memory + recent messages) |
| **Session** (current session) | Tool results, conversation | Session notes via `save_note` | `temp:` prefixed state | Short-term memory (vector) | Recall memory |
| **Persistent** (cross-session) | `claude-progress.txt`, CLAUDE.md, MEMORY.md | YAML frontmatter, consolidated global memories | `app:` and `user:` prefixed state; Memory Bank | Long-term memory (SQLite), User memory | Archival memory (vector) |
| **Entity** (relationship tracking) | — | — | — | Entity memory (graph) | — |

### 5.2 Memory Lifecycle

**OpenAI Four-Phase Pipeline**:
```
Inject → Distill → Consolidate → Reinject
```

**Anthropic**:
```
Load (CLAUDE.md + progress file) → Work → Compact → Save (progress file + git commit)
```

**MemGPT**:
```
Compile context → Work → Self-manage (insert/search/replace/delete) → Persist
```

**CrewAI**:
```
Load (all 4 memory types) → RAG search per step → Accumulate → Persist
```

---

## 6. Caching Strategies

| Vendor | Caching Mechanism | Key Details |
|--------|------------------|-------------|
| **Anthropic** | Prompt caching with `cache_control` | Up to 4 breakpoints; tools → system → messages order; 1,024 token minimum; 5-min TTL (1-hour for newer models) |
| **OpenAI** | Automatic caching | 50-90% discount; static content first, variable last; Responses API boosts utilization to 80% |
| **Google/ADK** | `ContextCacheConfig` + Gemini cached content API | Fingerprint-based; created on second matching request; 30-min TTL; strips cached content from subsequent requests |
| **Cursor** | Speculative edits (pre-computed completions) | Pre-computes likely next edits; caches for instant delivery |

---

## 7. Retrieval Strategies

| Vendor | Retrieval Approach | Key Innovation |
|--------|-------------------|----------------|
| **Anthropic** | Just-in-Time via tools | Lightweight references → load on demand; 95% context reduction |
| **OpenAI** | Tool-based retrieval + tool search for lazy loading | Deferred tool definitions loaded only when needed |
| **Google/ADK** | Memory services (InMemory, Vertex RAG, Memory Bank) | Memory Bank extracts "facts" from conversations rather than storing raw events |
| **Cursor** | Embedding → cross-encoder reranking | Two-stage retrieval dramatically improves precision over embedding alone |
| **CrewAI** | RAG across all 4 memory types | Each memory type has its own retrieval mechanism |
| **MemGPT** | Agent-initiated `memory_search` | Agent decides when and what to retrieve |

---

## 8. Unique Innovations by Vendor

| Vendor | Unique Innovation | Description |
|--------|------------------|-------------|
| **Anthropic** | Tool Result Clearing | Replace old tool results with placeholders; 84% token reduction |
| **Anthropic** | Thinking Block Clearing | Remove old reasoning blocks, keep only recent |
| **OpenAI** | Encrypted compaction items | Opaque but semantically rich compressed context |
| **OpenAI** | `input_filter` on handoffs | Surgical context filtering during agent transfers |
| **Google/ADK** | `include_contents="none"` | Complete history stripping with state-only injection |
| **Google/ADK** | Three-scope state prefixes | `app:` / `user:` / `temp:` for visibility control |
| **Cursor** | Cross-encoder reranking | Second-stage precision filtering after embedding recall |
| **Cursor** | Speculative edits | Pre-computed context for instant delivery |
| **LangGraph** | Reducer-based state merging | Per-field merge strategies |
| **CrewAI** | Entity Memory | Relationship graph of entities, not just text storage |
| **MemGPT** | Self-managed memory | Agent uses tools to manage its own context window |
| **Devin** | Environment-as-memory | Filesystem/terminal as external storage |
| **AutoGen** | Head-and-Tail context | Keep first K + last N messages, drop middle |
| **DSPy** | Programmatic optimization | Auto-search for optimal context configuration |

---

## 9. Performance Data Comparison

| Metric | Vendor | Data |
|--------|--------|------|
| Context editing token reduction | Anthropic | **84%** |
| Memory tool + context editing improvement | Anthropic | **39%** over baseline |
| Multi-agent vs. single-agent | Anthropic | **90.2%** improvement on research tasks |
| Token usage → performance correlation | Anthropic | **80%** of performance variance explained |
| Multi-agent token overhead | Anthropic | **~15x** more tokens than single-agent |
| Just-in-Time context reduction | Anthropic | **95%** via tool lazy loading |
| Long context query placement improvement | Anthropic | Up to **30%** with queries at end |
| Prompt caching cost reduction | Anthropic | Up to **90%** |
| Prompt caching latency reduction | Anthropic | Up to **85%** |
| Chat → Responses API cache utilization | OpenAI | 40% → **80%** |
| Cached token cost reduction (o4-mini) | OpenAI | **75%** cheaper |
| Planning prompt improvement (SWE-bench) | OpenAI | **4%** pass rate increase |
| Reasoning performance degradation | OpenAI | Starts at ~**3,000 tokens** |
| Performance ceiling | Anthropic | ~**1M tokens** |

---

## 10. Decision Matrix: When to Use What

| Scenario | Recommended Approach | Inspired By |
|----------|---------------------|-------------|
| Simple single-agent chat | Clear system prompt + few-shot examples | Anthropic Level 1-3 |
| Multi-turn with growing history | Context trimming + summarization | OpenAI Phase 4, LangGraph |
| Multi-agent pipeline | State-based context passing + isolation | Google/ADK `output_key` + `include_contents="none"` |
| Code-aware context | AST chunking + embedding + reranking | Cursor two-stage retrieval |
| Long-running autonomous tasks | Progress files + git checkpoints + compaction | Anthropic harness pattern |
| Cross-session personalization | Four-phase memory pipeline | OpenAI inject → distill → consolidate → reinject |
| Entity-rich domains | Entity memory with relationship tracking | CrewAI entity memory |
| Unlimited context needs | Self-managed virtual memory | MemGPT/Letta |
| Cost-sensitive production | Prompt caching + model routing | Anthropic caching + model routing |
| Latency-sensitive features | Speculative pre-loading + custom small models | Cursor speculative edits |

---

## 11. The One Key Takeaway from Each

| Source | Most Valuable Insight |
|--------|----------------------|
| **Anthropic** | **Degradation chain** (clear → summarize → drop) — not a single strategy |
| **OpenAI** | **Four-phase memory pipeline** (inject → distill → consolidate → reinject) |
| **Google/ADK** | **`include_contents="none"` + state injection** — cleanest isolation |
| **Cursor** | **Two-stage retrieval** (embedding recall → cross-encoder rerank) |
| **LangGraph** | **Reducer pattern** — each state field has its own merge strategy |
| **CrewAI** | **Entity memory** — relationship graphs, not just text |
| **MemGPT** | **Self-managed memory** — LLM decides what to remember/forget |
| **Devin** | **Environment as memory** — don't remember everything, know where to find it |
| **AutoGen** | **Head-and-Tail** — anchor beginning + keep recent, drop middle |
| **DSPy** | **Programmatic optimization** — auto-search for optimal context config |
| **Academia** | **Lost in the Middle** — important info at start/end, not middle |

---

## References

See individual practice documents:
- [04_Anthropic_CE_Practices](04_Anthropic_CE_Practices.md)
- [05_OpenAI_CE_Practices](05_OpenAI_CE_Practices.md)
- [06_Google_CE_Practices](06_Google_CE_Practices.md)
- [07_Cursor_CE_Practices](07_Cursor_CE_Practices.md)
- [08_Frameworks_CE_Practices](08_Frameworks_CE_Practices.md)
