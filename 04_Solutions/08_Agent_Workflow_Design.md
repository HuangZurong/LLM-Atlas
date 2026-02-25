# Agent Workflow Design: Business Process Orchestration with LLMs

*Prerequisite: [../03_Engineering/05_Agent/](../03_Engineering/05_Agent/).*

---

This document covers how to design LLM-powered agent workflows that map to real business processes, moving beyond single-turn Q&A to multi-step, stateful orchestration.

## 1. From Chatbot to Workflow Engine

| Maturity Level | Pattern | Example |
| :--- | :--- | :--- |
| **L1: Q&A** | Single-turn RAG | "What is our refund policy?" |
| **L2: Multi-turn** | Conversational RAG with memory | "What about for international orders?" (follow-up) |
| **L3: Task Agent** | Tool-calling agent | "Process a refund for order #12345." |
| **L4: Workflow** | Multi-agent orchestration | "Review all pending refunds, approve those under $100, escalate the rest." |

Most enterprise value lives at L3-L4. This document focuses on designing for those levels.

---

## 2. Core Workflow Patterns

### 2.1 Sequential Pipeline (Chain)

```
[Intake Agent] → [Research Agent] → [Draft Agent] → [Review Agent] → [Output]
```

- **Use Case**: Report generation, document drafting.
- **Pros**: Simple, predictable, easy to debug.
- **Cons**: Slow (no parallelism), single point of failure at each step.

### 2.2 Parallel Fan-Out / Fan-In

```
                 ┌→ [Source A Agent] ─┐
[Coordinator] ───┼→ [Source B Agent] ─┼→ [Synthesizer] → [Output]
                 └→ [Source C Agent] ─┘
```

- **Use Case**: Multi-source research, competitive analysis.
- **Pros**: Fast (parallel execution), comprehensive coverage.
- **Cons**: Requires a strong synthesizer to merge potentially conflicting results.

### 2.3 Router / Dispatcher

```
              ┌→ [Technical Support Agent]
[Router] ─────┼→ [Billing Agent]
              └→ [Sales Agent]
```

- **Use Case**: Customer service triage, intent-based routing.
- **Implementation**: LLM classifier or embedding-based similarity to route to the correct specialist agent.

### 2.4 Supervisor / Hierarchical

```
[Supervisor Agent]
    ├── assigns tasks to → [Worker Agent 1]
    ├── assigns tasks to → [Worker Agent 2]
    └── reviews outputs from all workers → [Final Decision]
```

- **Use Case**: Complex project management, code review pipelines.
- **Key**: The Supervisor has a "plan" and delegates sub-tasks. Workers report back. Supervisor synthesizes.

### 2.5 Handoff (Swarm Pattern)

```
[Agent A] ──handoff──→ [Agent B] ──handoff──→ [Agent C]
```

- **Use Case**: Multi-department processes (Sales → Legal → Finance).
- **Key**: Each agent has its own system prompt, tools, and context. The "handoff" transfers the conversation state.

---

## 3. State Management

Stateful workflows require persistent state across agent turns and even across sessions.

### 3.1 State Schema Design

```python
@dataclass
class WorkflowState:
    # Identity
    session_id: str
    user_id: str

    # Progress
    current_step: str           # e.g., "research", "drafting", "review"
    completed_steps: List[str]

    # Accumulated Knowledge
    context: Dict[str, Any]     # Retrieved docs, intermediate results
    messages: List[Message]     # Full conversation history

    # Control
    requires_approval: bool     # HITL gate
    error_count: int            # For retry/circuit-breaker logic
```

### 3.2 Persistence Options

| Option | Latency | Durability | Best For |
| :--- | :--- | :--- | :--- |
| **In-memory dict** | <1ms | None (lost on crash) | Prototyping |
| **Redis** | ~1ms | Configurable | Production real-time workflows |
| **PostgreSQL** | ~5ms | Full ACID | Audit-critical workflows |
| **LangGraph Checkpointer** | ~2ms | Backend-dependent | LangGraph-native apps |

---

## 4. Error Handling & Resilience

### 4.1 The "3R" Pattern

1. **Retry**: On transient failures (rate limits, timeouts), retry with exponential backoff.
2. **Re-route**: If a specific tool or model fails, fall back to an alternative (e.g., GPT-4o → Claude → local model).
3. **Raise**: If retries and re-routes are exhausted, escalate to a human operator with full context.

### 4.2 Circuit Breaker for Tool Calls

```
Tool Call → Success? → Continue
              ↓ No
         Retry (max 3) → Success? → Continue
              ↓ No
         Fallback Model → Success? → Continue
              ↓ No
         HALT + Notify Human
```

---

## 5. Human-in-the-Loop (HITL) Integration Points

| Gate Type | Trigger | Example |
| :--- | :--- | :--- |
| **Approval Gate** | Before any destructive action (delete, send, pay). | "Agent wants to send email to client. Approve?" |
| **Review Gate** | Before publishing or committing a generated artifact. | "Draft report ready. Review before sending?" |
| **Escalation Gate** | When agent confidence is below threshold. | "I'm not sure about this tax regulation. Routing to expert." |
| **Audit Gate** | Periodic sampling of agent decisions for quality control. | "5% of all auto-approved refunds are flagged for human review." |

---

## 6. Observability for Agent Workflows

Every production workflow needs end-to-end tracing:

```
[Trace ID: abc-123]
├── Step 1: Router (latency: 120ms, tokens: 450)
│   └── Decision: route to "Technical Support"
├── Step 2: Retrieval (latency: 80ms, docs: 5)
│   └── Sources: manual_v3.pdf (page 12), faq.md
├── Step 3: Generation (latency: 1.2s, tokens: 890)
│   └── Confidence: 0.92
└── Step 4: Response delivered (total: 1.4s)
```

**Key Metrics:**
- **End-to-end latency** per workflow type.
- **Step success rate** (which step fails most?).
- **Tool call frequency** and error rate.
- **HITL trigger rate** (too high = agent is under-confident; too low = risky).

---

## Key References

1. **Yao et al. (2023)**: *ReAct: Synergizing Reasoning and Acting in Language Models*.
2. **Wang et al. (2024)**: *A Survey on Large Language Model based Autonomous Agents*.
