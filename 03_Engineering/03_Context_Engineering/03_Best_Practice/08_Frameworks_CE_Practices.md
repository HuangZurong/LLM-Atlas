# 08 · Agent Frameworks Context Engineering Practices

*Compiled from official documentation, source code, blog posts, and academic papers covering LangGraph, CrewAI, AutoGen, MemGPT/Letta, Devin, Vercel AI SDK, DSPy, Semantic Kernel, and Haystack.*

---

## 1. LangChain / LangGraph

### 1.1 Core Concept: Graph State as Context Carrier

LangGraph uses a graph-based state machine architecture where a shared `State` object (TypedDict or Pydantic model) is the primary context carrier between nodes.

```python
from typing import Annotated, TypedDict
from langgraph.graph import add_messages

class MyState(TypedDict):
    messages: Annotated[list, add_messages]  # append merge
    plan: str                                 # overwrite merge
    facts: Annotated[list, operator.add]     # list append
```

### 1.2 Reducers — Per-Field Merge Strategies

Every node reads from and writes to shared state. **Reducer functions** on state fields control how updates merge:

| Reducer | Behavior | Use Case |
|---------|----------|----------|
| `add_messages` | Append new messages, handle deduplication | Chat history |
| `operator.add` | Concatenate lists | Accumulating facts/findings |
| Default (no reducer) | Overwrite | Latest plan, current status |
| Custom function | Any logic | Conflict resolution, voting |

This is unique among frameworks — **each state field has its own merge strategy**, enabling fine-grained context flow control.

### 1.3 Checkpointing and Memory

| Feature | Description |
|---------|-------------|
| **Persistent checkpointing** | `MemorySaver`, `SqliteSaver`, `PostgresSaver` — saves full graph state at each super-step |
| **Time-travel debugging** | Replay and inspect any previous state |
| **Human-in-the-loop** | Pause execution, modify state, resume |
| **Thread-level memory** | Persists within a single conversation thread |
| **Cross-thread memory (Store API)** | LangGraph Platform `Store` for persisting across conversations — user preferences, facts, learned behaviors |

### 1.4 Context Window Management Utilities

| Utility | Description |
|---------|-------------|
| `trim_messages()` | Keep only the last N messages or tokens |
| **Summarization node** | A node that summarizes older messages and replaces them with a summary |
| **Filtering** | Selectively remove messages by type (e.g., remove stale tool results) |

### 1.5 LangChain's Context Engineering Taxonomy

LangChain's blog post (June 2025) identified five context primitives:

1. **Instruction context**: System prompts, rules, guidelines
2. **Knowledge context**: Retrieved documents, tool outputs, search results
3. **Conversation context**: Message history, trimmed/summarized
4. **Structured output context**: Schemas, examples that guide output format
5. **Tool context**: Available tools and their descriptions

---

## 2. CrewAI

### 2.1 Four-Layer Memory Architecture

CrewAI has one of the most explicitly structured memory systems among agent frameworks:

| Memory Type | Storage | Scope | Description |
|-------------|---------|-------|-------------|
| **Short-Term Memory** | Vector store (RAG) | Current execution | Stores information from current crew run; automatically populated from agent interactions; searchable via embeddings |
| **Long-Term Memory** | SQLite | Cross-execution | Persists task results and learned insights; enables agents to improve over time |
| **Entity Memory** | Vector store + graph | Current execution | Tracks entities (people, organizations, concepts); builds a knowledge graph of relationships |
| **User Memory** | Persistent store | Cross-session | Stores user-specific preferences and history |

### 2.2 Context Assembly Per Agent Step

Before each agent action, CrewAI assembles context from:

```
1. Task description and expected output
2. Relevant short-term memories (RAG search)
3. Relevant long-term memories
4. Entity information
5. Results from upstream tasks (task dependencies)
6. Tool outputs from previous steps
```

### 2.3 Knowledge Sources

CrewAI supports injecting external knowledge (PDFs, text files, JSON) that gets chunked, embedded, and made available via RAG. Knowledge can be scoped to specific agents or shared across the crew.

### 2.4 Key Insight: Entity Memory

Entity Memory is unique to CrewAI — it doesn't just store text, it **builds a relationship graph** of entities encountered during execution. This is valuable for scenarios that need to track relationships between multiple objects (e.g., "which customer ordered which product from which supplier").

---

## 3. Microsoft AutoGen

### 3.1 ChatCompletionContext Protocol (AutoGen 0.4+)

AutoGen 0.4 introduced a formal abstraction for managing what goes into the LLM context window:

| Implementation | Strategy | Description |
|---------------|----------|-------------|
| `BufferedChatCompletionContext` | Sliding window | Keeps only the last N messages |
| `HeadAndTailChatCompletionContext` | Anchor + recency | Keeps the first K messages (system context, initial instructions) AND the last N messages (recent context); drops the middle |
| Custom implementation | Any | Implement the protocol for domain-specific strategies |

### 3.2 Head-and-Tail Pattern

This is a notable pattern not seen in other frameworks:

```
[System message + first K messages]  ← Anchored (always kept)
[... middle messages dropped ...]     ← Forgotten
[Last N messages]                     ← Recent (always kept)
```

Rationale: The beginning of a conversation often contains critical setup context (task description, constraints, examples) that should never be dropped, while the most recent messages contain the current working state.

### 3.3 Multi-Agent Context Sharing

| Pattern | Description |
|---------|-------------|
| **Group Chat Manager** | Orchestrates which agent speaks next; maintains shared message history |
| **Selector Group Chat** | Uses an LLM to decide which agent should respond based on current context |
| **Swarm pattern** | Agents hand off to each other with context transfer |
| **Handoffs** | Agents transfer relevant context to the next agent |

---

## 4. MemGPT / Letta

### 4.1 Core Concept: Virtual Memory Management

MemGPT treats the LLM context window like an OS manages virtual memory:

```
┌─────────────────────────────┐
│ Main Context (RAM)           │  ← Agent actively manages
│  - Core Memory (editable)    │    memory_insert / memory_replace
│  - Recent Messages           │
└─────────────────────────────┘
         ↕ page in / page out
┌─────────────────────────────┐
│ Archival Memory (Disk)       │  ← Vector store, unlimited capacity
│ Recall Memory (Disk)         │  ← Conversation history store
└─────────────────────────────┘
```

### 4.2 Self-Managed Memory

The key innovation: **the agent itself decides what to page in and out of its context window** using explicit memory tools:

| Tool | Description |
|------|-------------|
| `memory_insert` | Add new information to archival memory |
| `memory_search` | Search archival memory for relevant information |
| `memory_replace` | Update existing information in core memory |
| `memory_delete` | Remove information from memory |

The LLM is not just a consumer of context — it is an **active manager** of its own context window.

### 4.3 Context Compilation

MemGPT compiles the context window from multiple sources before each LLM call:

```
System prompt
+ Core memory blocks (editable persona, user info, etc.)
+ Relevant archival memory results (from search)
+ Recent message buffer
+ Current user message
= Compiled context window
```

### 4.4 Key Insight

This is the most radical approach to context engineering: instead of the developer pre-defining rules for what goes in and out of context, the **LLM itself learns to manage its own memory** through tool use. The tradeoff is that it requires additional LLM calls for memory management operations.

---

## 5. Devin (Cognition Labs)

### 5.1 Core Concept: Environment as Memory

Devin's key insight:

> **Don't try to keep everything in the context window. Instead, let the agent know where to find information in the environment.**

The filesystem, terminal, and browser serve as external memory that the agent can query on demand.

### 5.2 Architecture

| Component | Description |
|-----------|-------------|
| **Planner model** | Maintains high-level plan; holds strategic context |
| **Worker models** | Execute individual steps; get tactical context |
| **Scratchpad** | Persistent notes within a session — findings, decisions, intermediate results |
| **Event-based history** | Session history as a stream of events (commands, edits, browser actions) |

### 5.3 Context Assembly

For each LLM call, context is assembled from:

```
Current plan state
+ Relevant recent events
+ Scratchpad contents
+ Current environment state (file being edited, terminal output)
= Context for this step
```

### 5.4 Environment-as-Memory Pattern

| Instead of... | Devin does... |
|---------------|---------------|
| Keeping all file contents in context | `cat`/`grep` to re-read files when needed |
| Storing all command outputs in context | Re-running commands or checking terminal history |
| Remembering all search results | Re-searching when needed |
| Maintaining full conversation history | Writing key findings to scratchpad |

### 5.5 Key Insight

The environment itself is the largest external storage available. Rather than compressing information to fit in the context window, use tools to access the environment on demand. This is especially powerful for long-running tasks that span hours.

---

## 6. Vercel AI SDK

### 6.1 Core Message Handling

The AI SDK uses a `messages` array as the primary context carrier, following the OpenAI-compatible format: `{ role, content }` with support for tool calls and multi-modal content.

### 6.2 Context Management Utilities

| Utility | Description |
|---------|-------------|
| `trimMessages()` | Trim message history to fit within token limits; supports keeping system message + last N messages |
| `convertToLanguageModelMessage()` | Normalizes messages across different provider formats |
| Token counting utilities | Help manage context budgets across different models |

### 6.3 Multi-Step Tool Calls (Agentic Loops)

The `maxSteps` parameter enables agentic loops where the model can call tools and continue. Each step's tool results are automatically appended to the message context for the next step.

### 6.4 Provider-Agnostic Context

The AI SDK abstracts context management across providers (OpenAI, Anthropic, Google, etc.), handling provider-specific context format differences transparently.

---

## 7. DSPy (Stanford)

### 7.1 Programmatic Context Engineering

DSPy takes a fundamentally different approach — **programmatic optimization** of what context to include:

| Concept | Description |
|---------|-------------|
| **Signatures** | Define input/output fields that implicitly specify what context is needed |
| **Modules** | Composable units that manage their own context assembly |
| **Optimizers (Teleprompters)** | Automatically optimize what context (few-shot examples, instructions) to include for best performance |
| **Assertions** | Runtime checks that can trigger context modification and retry |

### 7.2 Key Insight

Instead of manually engineering context, DSPy **automatically searches** for the optimal context configuration (which examples, which instructions, which retrieval parameters) through optimization. This is the most "machine learning" approach to context engineering.

---

## 8. Microsoft Semantic Kernel

### 8.1 Context Management

| Feature | Description |
|---------|-------------|
| **KernelArguments** | Pass context between functions/plugins |
| **Chat History management** | Built-in token-aware truncation |
| **Memory connectors** | Integration with vector stores for RAG-based context |
| **Planner** | Generates plans that include context requirements for each step |

---

## 9. Haystack (deepset)

### 9.1 Pipeline-Based Context Management

| Feature | Description |
|---------|-------------|
| **DocumentStores** | Backend-agnostic storage for context documents |
| **Retrievers** | Multiple strategies: BM25, embedding, hybrid |
| **PromptBuilder** | Template-based context assembly with Jinja2 |
| **Pipeline branching** | Different context assembly paths based on query type |

---

## 10. Cross-Framework Patterns

### 10.1 Emerging Patterns

| Pattern | Description | Used By |
|---------|-------------|---------|
| **Hierarchical Memory** | Working/short-term/long-term memory tiers | CrewAI, MemGPT/Letta, LangGraph |
| **Self-Managed Memory** | Agent decides what to store/retrieve | MemGPT/Letta, Devin |
| **Context Compilation** | Assembling context from multiple sources into a coherent prompt | All frameworks |
| **Sliding Window + Anchors** | Keep first/last messages, drop middle | AutoGen, Vercel AI SDK |
| **RAG-Augmented Context** | Retrieve relevant documents/code at decision time | LlamaIndex, Haystack, CrewAI |
| **Environment-as-Memory** | Use tools to re-access info instead of storing in context | Devin, Copilot Agent, Windsurf |
| **Graph-Based Context Flow** | Explicit state graphs controlling context propagation | LangGraph, Semantic Kernel |
| **Programmatic Optimization** | Auto-tune context composition for performance | DSPy |
| **Reducer-Based Merging** | Per-field merge strategies for state updates | LangGraph |
| **Entity Tracking** | Build relationship graphs of entities in context | CrewAI |

### 10.2 Convergent Themes

1. **Context engineering > prompt engineering**: The industry consensus is shifting from "write a good prompt" to "assemble the right context dynamically"

2. **Hierarchical memory is becoming standard**: Nearly every serious agent framework now implements tiered memory (working memory in context window, short-term session memory, long-term persistent memory)

3. **The agent should manage its own context**: The MemGPT insight — that the LLM itself should decide what to page in and out — is increasingly adopted

4. **Code-specific context engineering is distinct**: AST-aware indexing, dependency graph traversal, and symbol resolution go beyond generic text RAG

5. **Environment as external memory**: Modern agents use their tools (file system, search, browser) as external memory they can query on demand

6. **Token budget management is a first-class concern**: Every framework now provides utilities for trimming, summarizing, or otherwise managing the token budget

---

## 11. Academic Foundations

### 11.1 Key Papers

| Paper | Key Finding | CE Implication |
|-------|-------------|----------------|
| **"Lost in the Middle"** (Liu et al., 2023) | LLMs perform worse when relevant info is in the middle of long contexts; attend most to beginning and end | Place important information at start or end of prompt |
| **"MemGPT"** (Packer et al., 2023) | Treat context window like OS virtual memory with hierarchical storage | Self-directed memory retrieval, context compilation, memory consolidation |
| **"Reflexion"** (Shinn et al., 2023) | Agents that reflect on failures and store reflections as context for future attempts | Reflection itself is compressed, high-value context |
| **"Voyager"** (Wang et al., 2023) | Minecraft agent builds a skill library — learned skills (code) stored and retrieved as context | Curriculum-driven context generation |
| **RAG Survey** (Gao et al., 2024) | Categorized RAG into Naive, Advanced, and Modular | Query rewriting, HyDE, iterative retrieval, adaptive retrieval |
| **Context Length Extension Survey** (2024) | RoPE scaling, ALiBi, landmark attention techniques | Longer windows change engineering tradeoffs |

### 11.2 Academic Taxonomy

The academic community consistently identifies three memory tiers for LLM agents:

```
Working Memory    → Context window (limited, fast)
Short-Term Memory → Current session store (medium, session-scoped)
Long-Term Memory  → Persistent store (unlimited, cross-session)
```

Planning and reflection modules compress experiences into reusable context, creating a feedback loop that improves context quality over time.

---

## References

- [LangGraph Memory Concepts](https://langchain-ai.github.io/langgraph/concepts/memory/)
- [LangChain Context Engineering Blog](https://blog.langchain.dev/context-engineering-for-agents/)
- [CrewAI Memory Documentation](https://docs.crewai.com/concepts/memory)
- [Microsoft AutoGen Documentation](https://microsoft.github.io/autogen/)
- [MemGPT / Letta Documentation](https://docs.letta.com/)
- [Vercel AI SDK Documentation](https://sdk.vercel.ai/docs/)
- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [Microsoft Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/)
- [Haystack Documentation](https://docs.haystack.deepset.ai/)
- Liu et al., "Lost in the Middle" (2023)
- Packer et al., "MemGPT: Towards LLMs as Operating Systems" (2023)
- Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning" (2023)
- Wang et al., "Voyager: An Open-Ended Embodied Agent with Large Language Models" (2023)
