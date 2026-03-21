# 06 · Google Context Engineering Practices

*Compiled from Google ADK source code, Gemini API documentation, Vertex AI docs, Google I/O 2025 announcements, and ADK sample agents.*

---

## 1. Core Philosophy

### 1.1 State as the Context Bus

Google ADK's central design principle: **structured state passing > conversation history**. Rather than relying on the raw message thread to carry information between agents, ADK uses a shared `session.state` dictionary as the primary context transport mechanism.

### 1.2 Long Context as a Feature

Unlike Anthropic and OpenAI which emphasize minimizing context, Google leans into Gemini's 1M+ token context window as a differentiator. Their guidance often recommends feeding entire documents or codebases into context rather than chunking for RAG — but with careful placement strategies.

---

## 2. ADK Context Architecture

### 2.1 Context Object Hierarchy

ADK implements a layered context system with four distinct objects, each providing different levels of access:

```
ReadonlyContext (base)
  └── CallbackContext (adds mutable state, artifacts, credentials)
        └── ToolContext (adds function_call_id, auth, search_memory)

InvocationContext (master context for a single invocation)
  └── References to all services: session, artifact, memory, credential
```

| Context Object | Access Level | Key Capabilities |
|---------------|-------------|------------------|
| `ReadonlyContext` | Read-only | `user_content`, `invocation_id`, `agent_name`, `state` (immutable `MappingProxyType`), `session` |
| `CallbackContext` | Read-write | Mutable state (via delta tracking), artifact load/save/list, credential management |
| `ToolContext` | Read-write + tools | `function_call_id`, auth request/response, tool confirmation, `search_memory()` |
| `InvocationContext` | Full | All services, agent states, branching info, cost tracking, resumability config |

### 2.2 The Three-Prefix State Convention

The `State` class implements a delta-tracking dict with three key prefixes that define scope:

```python
class State:
    APP_PREFIX = "app:"    # Shared across ALL sessions for the app
    USER_PREFIX = "user:"  # Shared across all sessions for a specific user
    TEMP_PREFIX = "temp:"  # Temporary, NOT persisted across invocations
```

State changes are tracked via a `_delta` dict. When you write `ctx.state['key'] = value`, both the live value and the delta are updated. The delta is committed to storage at the end of the invocation.

**Usage examples from ADK samples**:

```python
# FOMC Research agent — explicit state storage tool
def store_state_tool(state: dict, tool_context: ToolContext) -> dict:
    tool_context.state.update(state)
    return {"status": "ok"}

# Deep Search agent — accumulating sources across iterations
callback_context.state["url_to_short_id"] = url_to_short_id
callback_context.state["sources"] = sources

# Customer Service agent — pre-loading customer profile into session state
# (loaded before conversation starts, agent reads from state)
```

---

## 3. The `output_key` Pattern — Inter-Agent Context Passing

This is Google's primary mechanism for context flow between agents in multi-agent systems.

### 3.1 How It Works

When an agent has `output_key="some_key"`, its final text response is automatically saved to `session.state["some_key"]`. Downstream agents reference this via `{some_key}` template syntax in their instructions.

From the LlmAgent source:

```python
if self.output_key and event.is_final_response() and event.content and event.content.parts:
    result = ''.join(part.text for part in event.content.parts if part.text and not part.thought)
    if self.output_schema:
        result = self.output_schema.model_validate_json(result).model_dump(exclude_none=True)
    event.actions.state_delta[self.output_key] = result
```

### 3.2 Real Examples

**Story Teller agent** — state-machine for collaborative writing:

```python
prompt_enhancer:   output_key = "enhanced_prompt"
creative_writer:   output_key = "creative_chapter_candidate"
focused_writer:    output_key = "focused_chapter_candidate"
critique_agent:    output_key = "current_story"
editor_agent:      output_key = "final_story"
```

**Deep Search agent** — chaining research through state:

```python
section_planner:          output_key = "report_sections"
section_researcher:       output_key = "section_research_findings"
research_evaluator:       output_key = "research_evaluation"
enhanced_search_executor: output_key = "section_research_findings"  # Overwrites!
report_composer:          output_key = "final_cited_report"
```

**Parallel Task Decomposition** — fan-out/fan-in:

```
message_enhancer → output_key = "enhanced_message"
  ↓ (flows into all three parallel branches via state)
email_drafter    → output_key = "drafted_email"
slack_drafter    → output_key = "drafted_slack_message"
event_extractor  → output_key = "event_details"
  ↓ (all results available to summary_agent)
summary_agent reads all output_keys
```

---

## 4. Context Filtering and Windowing

### 4.1 ContextFilterPlugin

The `ContextFilterPlugin` provides two mechanisms for managing context window size:

```python
class ContextFilterPlugin(BasePlugin):
    def __init__(self, num_invocations_to_keep=None, custom_filter=None):
```

- `num_invocations_to_keep`: Keeps only the last N invocations (user message + model response pairs), trimming older history
- `custom_filter`: A callable `(List[Event]) -> List[Event]` for arbitrary context filtering logic

This runs as a `before_model_callback`, modifying `llm_request.contents` before it reaches the model.

### 4.2 `include_contents` — The Nuclear Option

The `include_contents` parameter on LlmAgent:

```python
include_contents: Literal['default', 'none'] = 'default'
# 'none': Model receives NO prior history, operates solely on
#         current instruction and input
```

**This is the cleanest context isolation pattern in any framework.** The Deep Search agent uses it on the report_composer:

```python
report_composer = LlmAgent(
    include_contents="none",  # No conversation history
    instruction="""
    Research Plan: {research_plan}
    Research Findings: {section_research_findings}
    Citation Sources: {sources}
    Report Structure: {report_sections}
    """,
)
```

Strip conversation history entirely and inject only structured data via state template variables.

### 4.3 Branching for Context Isolation

The `InvocationContext.branch` field provides context isolation between sub-agents:

```python
branch: Optional[str] = None
# Format: agent_1.agent_2.agent_3
# Used when multiple sub-agents shouldn't see their peer agents'
# conversation history
```

Events can be filtered by branch via `_get_events(current_branch=True)`, ensuring parallel sub-agents don't pollute each other's context windows.

---

## 5. Context Caching (Gemini-Specific)

### 5.1 Configuration

```python
class ContextCacheConfig(BaseModel):
    cache_intervals: int = 10    # Max invocations to reuse same cache
    ttl_seconds: int = 1800      # 30 minutes TTL
    min_tokens: int = 0          # Minimum tokens to trigger caching
```

### 5.2 Caching Strategy

1. **First request**: Generate a fingerprint (hash of system instruction + tools + first N contents) but don't create a cache yet
2. **Second request**: If fingerprint matches, create a Gemini cached content object via `genai_client.aio.caches.create()`
3. **Subsequent requests**: Validate the cache (not expired, not exceeded interval limit, fingerprint still matches). If valid, strip system instruction/tools/cached contents from the request and set `cached_content = cache_name`
4. **Cache boundary**: Determined by finding the last continuous batch of user contents and caching everything before it

This is a significant cost optimization for agents with large, stable system instructions and tool definitions.

---

## 6. Memory Services — Long-Term Context

ADK provides three memory service implementations:

### 6.1 InMemoryMemoryService

Keyword-matching for prototyping. Stores session events in memory, searches by word overlap.

### 6.2 VertexAiRagMemoryService

Uses Vertex AI RAG corpus:
- Uploads session events as JSON lines to a RAG corpus
- Retrieves via `rag.retrieval_query()` with configurable `similarity_top_k` and `vector_distance_threshold`

### 6.3 VertexAiMemoryBankService

Uses Vertex AI Memory Bank (managed service):
- Generates structured memories from session events via `memories.generate()`
- Retrieves via `memories.retrieve()` with similarity search
- **Extracts "facts" from conversations** rather than storing raw events

### 6.4 Tool-Level Memory Access

Tools can search memory via:

```python
results = tool_context.search_memory(query)
# Returns MemoryEntry objects with content, author, and timestamp
```

---

## 7. Grounding and Context Augmentation

### 7.1 The Grounding-to-Citation Pipeline

From the Deep Search agent, Google's complete grounding pattern:

1. `google_search` tool provides web grounding with `grounding_metadata` on events
2. `grounding_chunks` contain web source URIs, titles, and domains
3. `grounding_supports` contain text segments with confidence scores mapped to chunks
4. `collect_research_sources_callback` aggregates these into a structured citation database in state
5. `citation_replacement_callback` post-processes the final report to replace `<cite source="src-N"/>` tags with markdown links

This is a complete grounding-to-citation pipeline implemented entirely as context engineering callbacks.

---

## 8. Multi-Agent Context Orchestration

### 8.1 Four Orchestration Primitives

Each has distinct context flow semantics:

| Primitive | Context Flow | Use Case |
|-----------|-------------|----------|
| **SequentialAgent** | Agents run in order; each sees accumulated state from all previous agents | Pipelines where context builds up step by step |
| **ParallelAgent** | Agents run concurrently; each gets a snapshot of current state; outputs merged back | Independent tasks that can be done simultaneously |
| **LoopAgent** | Repeats sub-agents up to `max_iterations`; state accumulates across iterations | Generate-evaluate-improve cycles |
| **LLM-based transfer** | LlmAgent dynamically transfers to sub-agents or peers based on conversation | Dynamic routing based on user intent |

### 8.2 Transfer Control

Controlled by two parameters:
- `disallow_transfer_to_parent`: Prevents agent from escalating back
- `disallow_transfer_to_peers`: Prevents agent from transferring to sibling agents

---

## 9. Gemini Long Context Best Practices

### 9.1 Placement Strategies

| Strategy | Description |
|----------|-------------|
| Instructions at both beginning and end | Mitigates "lost in the middle" effect |
| Native multimodal context | Feed PDFs, images, audio, video directly rather than extracting text |
| Full document in-context | For retrieval tasks, Gemini can process entire documents in-context rather than chunked RAG, often with better accuracy |
| Many-shot prompting | Use long context for dozens to hundreds of examples |
| Full codebase in context | Feed entire codebases rather than individual files |

### 9.2 Context Caching for Cost Management

With 1M+ token contexts, caching becomes critical. Use `ContextCacheConfig` (Section 5) to avoid re-sending large stable prefixes.

---

## 10. Google I/O 2025 and Recent Developments

| Announcement | Description |
|-------------|-------------|
| Gemini 2.5 Pro | 1M token context with improved "needle in a haystack" performance |
| Agent-to-Agent (A2A) protocol | Cross-framework agent communication standard |
| ADK open-source release | Official agent framework with the context patterns described above |
| Vertex AI Agent Engine | Managed deployment with built-in session and memory management |
| Memory Bank | Managed service for long-term agent memory (VertexAiMemoryBankService) |

---

## 11. Key Context Engineering Patterns Summary

| Pattern | Description | ADK Mechanism |
|---------|-------------|---------------|
| **State as context bus** | Use `output_key` and `state[]` to pass structured data between agents | `output_key`, `{template}` syntax |
| **Context isolation** | Strip history and inject only what's needed via state templates | `include_contents="none"` |
| **Branching** | Branch-based event filtering to prevent context pollution | `InvocationContext.branch` |
| **Context windowing** | Manage context size with invocation limits or custom filters | `ContextFilterPlugin` |
| **Context caching** | Cache stable portions across invocations | `ContextCacheConfig` |
| **Memory services** | Long-term context that persists across sessions | RAG or Memory Bank |
| **Callback-based transformation** | Transform context before/after model calls | `before_model_callback`, `after_agent_callback` |
| **Grounding pipeline** | Google Search grounding with structured source tracking | Grounding metadata + callbacks |
| **Delta-tracked state** | All state mutations tracked as deltas for efficient persistence | `State._delta` |
| **Three-scope prefixes** | Control state visibility and persistence scope | `app:`, `user:`, `temp:` |

---

## References

- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [Google ADK Samples Repository](https://github.com/google/adk-samples)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [Vertex AI Agent Engine](https://cloud.google.com/vertex-ai/docs/agents)
- [Gemini Long Context Guide](https://ai.google.dev/gemini-api/docs/long-context)
- [Google I/O 2025 Developer Keynote](https://io.google/2025/)
- ADK source code: `google.adk.agents`, `google.adk.sessions.state`, `google.adk.plugins.context_filter_plugin`, `google.adk.models.gemini_context_cache_manager`
