# 05 · OpenAI Context Engineering Practices

*Compiled from OpenAI's official API documentation, Cookbook entries, Agents SDK docs, GPT-4.1/5/5.2 prompting guides, and public research.*

---

## 1. Core Philosophy

### 1.1 The Operating System Metaphor

> The LLM is a CPU, the context window is RAM, and the developer's job is to be the operating system — loading working memory with exactly the right code and data for each task. — Andrej Karpathy (June 2025)

### 1.2 Guiding Principle

Given a finite attention budget, find the **smallest possible set of high-signal tokens** that maximize the likelihood of the desired outcome. Research shows LLM reasoning performance starts degrading around 3,000 tokens — well below technical maximums. Even GPT-5's 272K input token window can be overwhelmed by uncurated histories, redundant tool results, or noisy retrievals.

---

## 2. The Four Context Engineering Strategies

| Strategy | Description | OpenAI Implementation |
|----------|-------------|----------------------|
| **Write** | Persist context externally (scratchpads, notes, memory stores) | `RunContextWrapper` state objects, memory distillation tools |
| **Select** | Retrieve only what is relevant (RAG, semantic search) | Dynamic instructions, tool-based retrieval |
| **Compress** | Retain only the tokens required (summarize, compact) | Compaction API, session compression, context summarization |
| **Isolate** | Split context across agents with separate windows | Handoffs, agents-as-tools, `input_filter` on handoffs |

---

## 3. System Message / Instruction Design

### 3.1 General Principles

- Be **clear, specific, and unambiguous** — most prompt failures come from ambiguity, not model limitations
- Avoid conflicting instructions (e.g., "be brief" and "be comprehensive" without prioritization)
- Avoid overly long system messages — they consume context window and reduce room for user content
- If output format matters, state it explicitly
- Pin production applications to specific model snapshots (e.g., `gpt-4.1-2025-04-14`)

### 3.2 GPT-4.1 Agentic System Prompt Structure

Three key reminders for all agent system prompts:

1. **Persistence**: "You are an agent — please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user."
2. **Tool-Calling**: "If you're unsure, use your tools to read files, search, or verify before answering."
3. **Planning**: "Think step-by-step before every tool call. Reflect after each one. Plan, act, and verify before responding." (Boosted SWE-bench pass rates by 4%.)

### 3.3 GPT-5.2 CTCO Pattern

The recommended prompt structure for GPT-5.2:

```
C — Context: Background information
T — Task: What to do
C — Constraints: Rules and boundaries
O — Output: Expected format
```

Strip "personality" padding like "Take a deep breath" or "You are a world-class expert" — GPT-5.2 treats this as noise.

### 3.4 Long Context Instruction Placement

For long-context prompts, place instructions at **both the beginning and end** of the provided context. If placing only once, above the context works better than below.

### 3.5 Dynamic Instructions Pattern (Agents SDK)

```python
def dynamic_instructions(context: RunContextWrapper, agent: Agent) -> str:
    return f"The user's name is {context.context.user_name}. Today is {date.today()}."

agent = Agent(
    name="my_agent",
    instructions=dynamic_instructions,  # function, not static string
)
```

---

## 4. Conversation State Management

### 4.1 Responses API: Two Approaches

| Approach | Description |
|----------|-------------|
| **Automatic (Server-Side)** | Pass `previous_response_id` to chain responses; OpenAI manages history. All previous input tokens are still billed |
| **Manual (Client-Side)** | Collect outputs into a list and resubmit as input for the next response; gives full control |

### 4.2 Critical Rule for Reasoning Models

When tool use is involved with reasoning models (o3, o4-mini), you **must** include reasoning items in conversation history — either via `previous_response_id` or by explicitly adding reasoning items to input. Omitting them degrades performance.

---

## 5. Context Compaction

### 5.1 Two Modes

**Server-Side (Automatic)**:
```python
response = client.responses.create(
    model="gpt-5",
    input=[...],
    context_management={
        "compact_threshold": 50000  # trigger when token count crosses this
    }
)
```

**Standalone (Explicit)**: Call `/responses/compact` endpoint directly. Send the full window; receive a compacted window back.

### 5.2 How It Works

- All prior **user messages are kept verbatim**
- Prior assistant messages, tool calls, tool results, and encrypted reasoning are replaced with a **single encrypted compaction item**
- The compaction item preserves the model's latent understanding while remaining opaque and ZDR-compatible
- After compaction, you can drop items that came before the most recent compaction item

### 5.3 Chaining After Compaction

Two patterns:
1. **Stateless input-array chaining**: Append output (including compaction items) to your next input array
2. **`previous_response_id` chaining**: Pass only the new user message each turn and carry the ID forward

---

## 6. Agents SDK Context Management

### 6.1 RunContextWrapper — Dependency Injection

```python
@dataclass
class MyContext:
    user_name: str
    user_id: str
    logger: Logger
    db: DatabaseConnection

context = MyContext(user_name="Alice", user_id="123", logger=logger, db=db)
result = await Runner.run(agent, input="Hello", context=context)
```

Key rules:
- The context object is **not sent to the LLM** — it is purely local for dependency injection
- Every agent, tool, handoff, and lifecycle hook in a given run must use the **same context type**
- Use it for: user metadata, loggers, database connections, feature flags, etc.

### 6.2 Session-Based Memory (Short-Term)

Two techniques:

| Technique | Pros | Cons |
|-----------|------|------|
| **Context Trimming** (keep last N turns) | Deterministic, zero added latency, keeps recent work verbatim | Abruptly forgets long-range context; important constraints can vanish |
| **Context Compression** (summarize older turns) | Preserves long-range context, corrects prior mistakes ("clean room" effect) | Adds latency (extra model call), potential summarizer variability |

### 6.3 Making Data Available to the LLM

Since `RunContextWrapper` is not visible to the LLM, you must use one of:
1. **Agent instructions** (system prompt) — static strings or dynamic functions
2. **Tool outputs** — return data from tools
3. **Conversation history** — inject into prior messages

---

## 7. Memory Management Pipeline

OpenAI's recommended four-phase memory pipeline:

### Phase 1: Memory Injection

Inject only relevant portions of state into context at session start:
- Use **YAML frontmatter** for structured, machine-readable metadata
- Use **Markdown notes** for flexible, human-readable memory
- State-based memory > retrieval-based memory (structured fields with clear precedence vs. brittle similarity search)

### Phase 2: Memory Distillation

Capture dynamic insights during active turns via a dedicated tool (e.g., `save_note` tool). The agent writes session notes as it learns new things.

### Phase 3: Memory Consolidation

After the session, merge session-level notes into a dense, conflict-free set of global memories:
- Prune stale, overwritten, or low-signal memories
- Deduplicate aggressively over time
- Two-phase processing (note-taking then consolidation) is more reliable than building the whole memory system at once

### Phase 4: Context Window Management

Keep only the last N user turns. When trimming occurs, reinject session-scoped memories into the system prompt on the next turn.

---

## 8. Multi-Agent Context Isolation

### 8.1 Two Patterns

| Pattern | Description | Use When |
|---------|-------------|----------|
| **Agents as Tools** (Manager Pattern) | Central agent invokes specialists as tools, retains control and context | Specialist should help with a bounded subtask but not take over |
| **Handoffs** (Decentralized) | One agent directly transfers control to another; target receives conversation history | Agent should own the next part of the interaction |

### 8.2 Context Isolation via `input_filter`

```python
from agents.extensions.handoff_filters import remove_all_tools

handoff = Handoff(
    agent=refund_agent,
    input_filter=remove_all_tools  # strips tool calls from history
)
```

### 8.3 Nested Handoff History (Beta)

When enabled via `RunConfig.nest_handoff_history`, the runner collapses the prior transcript into a single assistant summary wrapped in a `<CONVERSATION HISTORY>` block. Custom mapping functions via `RunConfig.handoff_history_mapper` are also supported.

---

## 9. Function Calling & Tool Result Management

### 9.1 Token Efficiency

- Function definitions are injected into the system message and **count against context limits / billed as input tokens**
- Reduce token usage by: limiting functions loaded upfront, shortening descriptions, using **tool search** so deferred tools load only when needed
- Fine-tuning can reduce tokens from function specifications

### 9.2 Tool Result Filtering

For data-intensive apps, filter tool call responses down to essential fields only. Avoid returning thousands of tokens of raw data when a few key fields suffice.

### 9.3 Structured Outputs

Use function calling for tool connections; use `response_format` for structured user-facing responses. JSON Schema definitions are converted to context-free grammars under the hood.

---

## 10. Caching for Cost and Latency Optimization

Structure prompts for caching:
- Place **static content first**: system instructions, few-shot examples, tool definitions
- Place **variable content last**: user messages, query-specific data
- OpenAI offers **automatic caching** with 50–90% discounts depending on the model
- Switching from Chat Completions to Responses API boosted cache utilization from 40% to 80% in tests
- Cached input tokens for o4-mini are **75% cheaper** than uncached ones

---

## 11. Security Considerations

- Put untrusted data in `untrusted_text` blocks when available
- Quoted text, multimodal data, file attachments, and tool outputs are assumed untrusted with no authority by default
- Without proper formatting, untrusted input may contain prompt injection attacks that are extremely difficult for the model to distinguish from developer instructions
- Avoid putting secrets in `RunContextWrapper.context` if you intend to persist or transmit serialized state

---

## 12. Evaluation-Driven Context Engineering

OpenAI repeatedly emphasizes: **"Evals is all you need for context engineering too."**

- AI engineering is inherently an empirical discipline; LLMs are inherently nondeterministic
- Build informative evals and iterate often
- Pin to specific model snapshots and build evals that measure prompt performance
- Start with `reasoning_effort="medium"` and tune based on eval results

---

## References

- [OpenAI Prompt Engineering Guide](https://developers.openai.com/api/docs/guides/prompt-engineering/) — Official API Docs
- [GPT-4.1 Prompting Guide](https://cookbook.openai.com/examples/gpt4-1_prompting_guide) — Agentic prompting best practices
- [GPT-5 Prompting Guide](https://cookbook.openai.com/examples/gpt-5/gpt-5_prompting_guide) — GPT-5 specific guidance
- [GPT-5.2 Prompting Guide](https://cookbook.openai.com/examples/gpt-5/gpt-5-2_prompting_guide) — CTCO pattern, enterprise workloads
- [Codex Prompting Guide](https://developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide) — Compaction for multi-hour reasoning
- [A Practical Guide to Building Agents (PDF)](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf) — OpenAI's 32-page agent building guide
- [Context Engineering for Personalization (Cookbook)](https://developers.openai.com/cookbook/examples/agents_sdk/context_personalization/) — Long-term memory patterns
- [Short-Term Memory Management with Sessions (Cookbook)](https://developers.openai.com/cookbook/examples/agents_sdk/session_memory/) — Trimming and compression
- [Agents SDK Context Management](https://openai.github.io/openai-agents-python/context/) — RunContextWrapper docs
- [Agents SDK Multi-Agent Orchestration](https://openai.github.io/openai-agents-python/multi_agent/) — Handoffs and agents-as-tools
- [Compaction Guide](https://developers.openai.com/api/docs/guides/compaction/) — Native context compaction API
- [Conversation State Guide](https://platform.openai.com/docs/guides/conversation-state) — Managing state across turns
- [Function Calling Guide](https://platform.openai.com/docs/guides/function-calling) — Tool definitions and token management
