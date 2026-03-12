# Memory Architecture Patterns: Decision Guide

*Prerequisite: [../01_Theory/01_Memory_Systems.md](../01_Theory/01_Memory_Systems.md).*

---

Choosing the right memory architecture is one of the most impactful decisions in LLM application design. This guide provides a decision framework based on real-world production scenarios.

## 1. The Decision Tree

```
Is the conversation multi-turn?
├── No → No memory needed. Stateless API call.
└── Yes
    ├── Single session only?
    │   ├── Short (<20 turns) → Buffer Memory (keep all turns)
    │   └── Long (>20 turns) → Hybrid Memory (Summary + Sliding Window)
    └── Cross-session persistence needed?
        ├── Structured facts (preferences, decisions)?
        │   └── Entity Memory (JSON/Graph DB)
        ├── Unstructured recall ("what did we discuss last week")?
        │   └── Vector Memory (Embedding + Retrieval)
        └── Both?
            └── Layered Architecture (Entity + Vector + Summary)
```

## 2. Architecture Comparison Matrix

| Architecture | Latency | Cost | Recall Quality | Complexity | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Buffer** | Lowest | High (full history in context) | Perfect (within window) | Trivial | Short conversations, prototyping |
| **Sliding Window** | Low | Medium | Good (recent only) | Low | Customer support bots |
| **Summary** | Medium (LLM call) | Low | Lossy | Medium | Long-running sessions |
| **Hybrid (Summary + Window)** | Medium | Medium | Good | Medium | Production chatbots |
| **Vector Retrieval** | Medium (embedding + search) | Low | Depends on embedding quality | High | Cross-session recall |
| **Entity** | High (extraction LLM call) | Medium | Structured, precise | High | Personalized assistants |
| **Layered (All)** | Highest | Highest | Best | Very High | Enterprise AI platforms |

## 3. Industrial Case Studies

### 3.1 ChatGPT Memory (OpenAI)
- **Architecture**: Entity Memory (extracted facts) + User-controlled CRUD.
- **Key Design**: User can view, edit, and delete individual memories.
- **Lesson**: Transparency and user control are non-negotiable for consumer products.

### 3.2 Claude Projects (Anthropic)
- **Architecture**: Project-level context injection (static documents).
- **Key Design**: No dynamic memory extraction; relies on user-curated context.
- **Lesson**: Sometimes "dumb" static context is more reliable than smart extraction.

### 3.3 Enterprise Assistants (Internal Tools)
- **Architecture**: Layered (Vector + Entity + Summary).
- **Key Design**: Separate memory stores per user, per project, per organization.
- **Lesson**: Multi-tenancy and access control are critical in enterprise.

## 4. Anti-Patterns

| Anti-Pattern | Problem | Fix |
| :--- | :--- | :--- |
| **Infinite Buffer** | Context window overflow, cost explosion | Set hard token budgets, use summary compression |
| **Blind Retrieval** | Injecting irrelevant memories that confuse the model | Use relevance threshold (e.g., cosine > 0.8) |
| **No Deduplication** | Same fact stored 50 times | Hash-based or semantic dedup before storage |
| **No Expiry** | Stale facts override current reality | Timestamp + decay scoring |
