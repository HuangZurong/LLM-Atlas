# Agentic Workflow Patterns

*Prerequisite: [02_Agent_Architecture.md](02_Agent_Architecture.md).*

---

Building reliable agentic systems means choosing the right orchestration pattern for the task. This document covers the five canonical patterns, their internal mechanics, trade-offs, and a decision framework for selection.

## 1. Prompt Chaining (Sequential Pipeline)

```
Input → LLM₁ → Gate → LLM₂ → Gate → LLM₃ → Output
```

- **Mechanism**: The output of step N becomes the input of step N+1. Each step has a focused, narrow instruction. A **Gate** (programmatic check or LLM judge) between steps validates quality before proceeding.
- **When to Use**: Tasks that decompose into fixed, predictable stages (e.g., Extract → Translate → Summarize → Format).
- **Strengths**: Deterministic flow, easy to debug (each step is independently testable), lowest token waste.
- **Weaknesses**: Rigid — cannot adapt if the task requires dynamic branching. Latency scales linearly with chain length.
- **Industrial Example**: Document processing pipelines (parse PDF → extract entities → generate summary → populate database).

## 2. Routing (Intent Classification)

```
Input → Classifier LLM → ┬─ Route A → Specialist Prompt A → Output
                          ├─ Route B → Specialist Prompt B → Output
                          └─ Route C → Specialist Prompt C → Output
```

- **Mechanism**: A lightweight classifier (small LLM or embedding-based) analyzes the input and dispatches it to a specialized handler. Each handler has its own optimized system prompt, tools, and even model.
- **When to Use**: Multi-functional systems where a single prompt would become bloated and unreliable (e.g., customer support routing to Billing vs. Tech Support vs. Returns).
- **Strengths**: Prevents "Prompt Bloat" — each route stays focused. Allows mixing models (cheap model for simple routes, expensive for complex).
- **Weaknesses**: Misrouting cascades into total failure. Requires a robust classifier and a fallback route.
- **Industrial Example**: Enterprise chatbots (Anthropic's Claude uses internal routing to select tool-use vs. direct-answer paths).

### Routing Implementation Strategies

| Strategy | Latency | Accuracy | Cost |
| :--- | :--- | :--- | :--- |
| **LLM Classification** (ask the model to pick a route) | ~500ms | High | Medium |
| **Embedding Similarity** (compare query to route descriptions) | ~50ms | Medium | Low |
| **Keyword/Regex Rules** | ~1ms | Low (brittle) | Free |
| **Hybrid** (rules first, LLM fallback) | ~50-500ms | High | Low-Medium |

## 3. Parallelization (Sectioning & Voting)

```
                    ┌─ Worker A ─┐
Input → Splitter ──>├─ Worker B ─┤──> Aggregator → Output
                    └─ Worker C ─┘
```

Two sub-patterns:

### 3.1 Sectioning (Map-Reduce)
- **Mechanism**: A large task is split into independent sub-tasks. Each sub-task is processed concurrently by a worker. Results are aggregated (concatenated, merged, or synthesized).
- **Example**: Summarizing a 100-page document — split into 10 sections, summarize each in parallel, then synthesize a final summary.

### 3.2 Voting (Ensemble)
- **Mechanism**: The same task is sent to N models (or the same model N times with temperature >0). A majority vote, consensus check, or judge selects the best answer.
- **Example**: Code generation — generate 5 solutions, run unit tests on each, pick the one that passes all tests.

- **Strengths**: Wall-clock time reduction (parallelism). Voting increases reliability for high-stakes decisions.
- **Weaknesses**: Token cost multiplied by N. Aggregation step can be non-trivial (conflicting sub-results).
- **Industrial Example**: Cursor/Copilot use parallel code generation + test-based selection.

## 4. Orchestrator-Workers (Hierarchical Delegation)

```
Input → Orchestrator LLM ──┬─ Assign Task 1 → Worker₁ → Result₁ ─┐
                            ├─ Assign Task 2 → Worker₂ → Result₂ ─┤──> Orchestrator → Synthesize → Output
                            └─ Assign Task 3 → Worker₃ → Result₃ ─┘
```

- **Mechanism**: A powerful "manager" LLM (e.g., Claude Opus, GPT-4o) dynamically decomposes the task, assigns sub-tasks to cheaper "worker" models, collects results, and synthesizes the final answer. Unlike Prompt Chaining, the plan is **dynamic** — the orchestrator decides the next step based on previous results.
- **When to Use**: Complex, open-ended tasks where the decomposition cannot be predetermined (e.g., "Research this topic and write a report").
- **Strengths**: Handles high complexity. Workers can be specialized (code agent, search agent, analysis agent).
- **Weaknesses**: Orchestrator is a single point of failure. High token cost for inter-agent communication. Requires robust state management.
- **Industrial Example**: Claude Code (Anthropic), Devin (Cognition), OpenAI Codex — all use an orchestrator that plans, delegates file edits/searches to workers, and synthesizes.

### State Management Patterns

| Pattern | Description | Use Case |
| :--- | :--- | :--- |
| **Message Passing** | Workers return results as messages to orchestrator | Simple delegation |
| **Shared Blackboard** | All agents read/write to a shared state object | Collaborative editing |
| **Event-Driven** | Agents emit events; orchestrator subscribes and reacts | Async, long-running tasks |

## 5. Evaluator-Optimizer (Self-Correction Loop)

```
Input → Generator → Draft → Evaluator → ┬─ Pass → Output
                                         └─ Fail → Feedback → Generator (Revise) → ...
```

- **Mechanism**: A generator produces a draft. An evaluator (separate LLM call, unit tests, or human) scores it against criteria. If it fails, structured feedback is sent back to the generator for revision. This loops until the output passes or a max iteration limit is reached.
- **When to Use**: Tasks where quality is paramount and can be objectively measured (code that must pass tests, compliance text that must meet legal criteria, translations that must preserve meaning).
- **Strengths**: Produces the highest quality output of any pattern. Self-improving within a single request.
- **Weaknesses**: Slowest and most expensive pattern (multiple round-trips). Risk of infinite loops if the evaluator and generator disagree.
- **Industrial Example**: AlphaCode (DeepMind) generates thousands of solutions, filters by test cases. Reflexion (Shinn et al.) uses verbal self-reflection as feedback.

### Evaluator Types

| Type | Speed | Reliability | Cost |
| :--- | :--- | :--- | :--- |
| **Programmatic** (unit tests, regex, schema validation) | Fast | Deterministic | Free |
| **LLM-as-Judge** (rubric-based scoring) | Slow | Probabilistic | Medium |
| **Human-in-the-Loop** | Very slow | Highest | Highest |
| **Hybrid** (programmatic gate + LLM judge for edge cases) | Medium | High | Low-Medium |

## 6. Pattern Selection Decision Framework

```
Start
  │
  ├─ Is the task decomposition fixed and predictable?
  │   └─ Yes → Prompt Chaining
  │
  ├─ Does the input need to go to different specialized handlers?
  │   └─ Yes → Routing
  │
  ├─ Can sub-tasks run independently in parallel?
  │   └─ Yes → Parallelization
  │
  ├─ Is the task open-ended, requiring dynamic planning?
  │   └─ Yes → Orchestrator-Workers
  │
  ├─ Is output quality critical and objectively measurable?
  │   └─ Yes → Evaluator-Optimizer
  │
  └─ Default → Start with Prompt Chaining (simplest), add complexity only when it fails.
```

## 7. Composition: Patterns Are Combinable

Production systems rarely use a single pattern. Common compositions:

| Composition | Example |
| :--- | :--- |
| **Routing + Chaining** | Route to the right handler, then chain through its pipeline |
| **Orchestrator + Parallelization** | Orchestrator assigns tasks, workers execute in parallel |
| **Chaining + Evaluator** | Each chain step has a quality gate (evaluator) before proceeding |
| **Routing + Orchestrator + Evaluator** | Route complex queries to an orchestrator that delegates and self-corrects |

**The golden rule**: Start with the simplest pattern that works. Add complexity only when you have evidence of failure on your actual query distribution.
