# LLM Memory Systems

*Prerequisite: [../../03_Context_Engineering/01_Theory/01_Context_Window_Mechanics.md](../../03_Context_Engineering/01_Theory/01_Context_Window_Mechanics.md).*
*See Also: [../../05_RAG/01_Theory/01_Architecture.md](../../05_RAG/01_Theory/01_Architecture.md) (RAG as external memory), [../../../02_Scientist/01_Architecture/12_Long_Context.md](../../../02_Scientist/01_Architecture/12_Long_Context.md) (long context theory).*

LLMs are stateless by default — each API call is independent with no memory of previous interactions. Memory systems bridge this gap, enabling continuity across turns and sessions.

---

## 1. Why Memory?

Without memory, every conversation starts from zero. Memory solves three problems:

| Problem | Without Memory | With Memory |
|---|---|---|
| **Continuity** | "What did I just ask?" — model doesn't know | Model recalls previous turns |
| **Personalization** | Same generic response for every user | Adapts to user preferences and history |
| **Context accumulation** | Must re-provide all context every time | Relevant context is retrieved automatically |

The fundamental constraint: **context windows are finite**. A 128K token window fills up quickly in multi-turn conversations with tool outputs. Memory is the engineering solution to this constraint.

## 2. Memory Taxonomy

### 2.1 Short-Term Memory (In-Context)

The conversation history within a single session, stored as the message array sent to the model.

```
messages = [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},       # turn 1
    {"role": "assistant", "content": "..."},   # turn 1
    {"role": "user", "content": "..."},       # turn 2
    {"role": "assistant", "content": "..."},   # turn 2
    ...  # grows with every turn
]
```

**Management strategies when context fills up**:

| Strategy | How | Tradeoff |
|---|---|---|
| **Truncation** | Drop oldest turns | Simple but loses early context |
| **Sliding window** | Keep last N turns | Predictable cost, loses long-range context |
| **Summarization** | LLM summarizes old turns into a compact paragraph | Preserves key info, but lossy |
| **Hybrid** | Summarize old turns + keep last N turns verbatim | Best balance of cost and quality |

### 2.2 Long-Term Memory (Persistent)

Cross-session memory that persists between conversations. Stored externally (database, vector store).

**Use cases**:
- User preferences ("I prefer Python over JavaScript")
- Facts learned in previous sessions ("User works at Company X")
- Project context ("We're building a RAG system for legal documents")

**Storage options**:

| Backend | Best For | Query Method |
|---|---|---|
| **Vector DB** (Pinecone, Qdrant, Chroma) | Semantic retrieval of past interactions | Embedding similarity search |
| **Key-Value Store** (Redis) | Fast lookup of structured facts | Exact key match |
| **Relational DB** (PostgreSQL) | Complex queries over structured memory | SQL |
| **Graph DB** (Neo4j) | Entity relationships | Graph traversal |

### 2.3 Working Memory (Scratchpad)

Temporary state during a single complex task. Not persisted across sessions.

- Agent scratchpads (intermediate reasoning steps)
- Tool call results waiting to be synthesized
- Partial outputs in a multi-step pipeline

## 3. Memory Architectures

### 3.1 Buffer Memory

The simplest approach: store the full conversation, truncate when it exceeds the token limit.

```
if token_count(messages) > MAX_TOKENS:
    messages = messages[-N:]  # keep last N messages
```

**Pros**: Zero information loss within the window.
**Cons**: Abrupt loss of context when truncation happens. Expensive for long conversations.

### 3.2 Summary Memory

Periodically compress the conversation into a running summary:

```
[Turn 1-10 Summary]: "User asked about RAG architecture. We discussed
 chunking strategies and decided on 512-token chunks with 50-token overlap."
[Turn 11]: (verbatim)
[Turn 12]: (verbatim)
```

**Pros**: Constant memory footprint regardless of conversation length.
**Cons**: Summarization is lossy — specific details (numbers, code snippets) may be lost.

### 3.3 Entity Memory

Extract and track named entities across the conversation:

```json
{
  "user": {"name": "Alice", "role": "ML Engineer", "preference": "Python"},
  "project": {"name": "LegalRAG", "stack": "LangChain + Qdrant", "status": "prototyping"},
  "decision": {"chunking": "512 tokens", "embedding": "BGE-M3"}
}
```

**Pros**: Structured, queryable, no redundancy.
**Cons**: Extraction is imperfect. Requires entity resolution (is "the project" the same as "LegalRAG"?).

### 3.4 Retrieval-Augmented Memory

Store all past interactions in a vector database. For each new query, retrieve the most relevant past context:

```
User: "What chunking strategy did we decide on?"
    ↓
Embed query → Search vector DB → Retrieve relevant past turns
    ↓
Inject retrieved context into the prompt
```

**Pros**: Scales to unlimited history. Only retrieves what's relevant.
**Cons**: Retrieval quality depends on embedding model. May miss context that's relevant but semantically distant.

## 4. Implementation Patterns

### 4.1 Memory as a System Component

```
User Query
    ↓
┌─────────────────────┐
│  Memory Manager      │
│  1. Read: retrieve   │ ← query long-term memory
│  2. Inject: add to   │ → add to prompt context
│     prompt context   │
│  3. Write: store     │ ← save new interactions
│     new interactions │
└─────────────────────┘
    ↓
LLM generates response
```

### 4.2 Memory as a Tool

In agent systems, memory can be exposed as a tool the agent decides when to use:

```
Tools available:
- memory_read(query): Search past conversations for relevant context
- memory_write(key, value): Store a fact for future reference
- memory_forget(key): Delete a stored fact
```

This gives the agent explicit control over what to remember and when to recall.

### 4.3 Framework Support

| Framework | Memory Types | Notes |
|---|---|---|
| **LangChain** | Buffer, Summary, Entity, VectorStore | Most comprehensive, modular |
| **LlamaIndex** | Chat memory + index-based retrieval | Tightly integrated with RAG |
| **Mem0** | Dedicated memory layer | Specialized for long-term user memory |
| **Custom** | Any combination | Full control, more engineering effort |

## 5. Challenges

### 5.1 Context Window vs Completeness

More memory context = better continuity, but also = higher cost and potential "lost in the middle" effects. The optimal amount of injected memory is task-dependent.

### 5.2 Stale Memory

Outdated information persists and may contradict current reality:
- "User prefers React" (but they switched to Vue 3 months ago)
- Solution: timestamp memories, decay old ones, allow explicit updates

### 5.3 Privacy and the Right to Forget

What should be remembered vs forgotten?
- Users may want to delete specific memories
- Regulatory requirements (GDPR right to erasure)
- Solution: explicit memory management APIs with user control

### 5.4 Consistency

Contradictory memories across sessions:
- Session 1: "Budget is $10K"
- Session 5: "Budget is $50K"
- Solution: version memories with timestamps, prefer most recent
