# 01 · Introduction to Context Engineering

*Prerequisite: [../../02_Prompt_Engineering/README.md](../../02_Prompt_Engineering/README.md).*
*Position in CE Pipeline: Orientation before the full `Load -> Budget -> Compress -> Assemble -> Observe` workflow.*

---

## 1. What Context Engineering Is

Anthropic describes context engineering as the natural progression of prompt engineering. In its September 29, 2025 engineering post, Anthropic frames it as the practice of curating and maintaining the optimal set of tokens during LLM inference, including information that comes from outside the prompt itself, such as tools, memory, external data, and message history.

In practical terms:

- **Prompt Engineering** asks: how should we write the instruction?
- **Context Engineering** asks: what information should the model see *right now*, in what order, at what priority, and in what form?

This module adopts that broader definition. A production LLM system rarely runs on a single prompt. It runs on a continuously assembled context made of:

- system instructions
- task-specific rules
- conversation history
- retrieved memory
- RAG results
- tool definitions
- tool outputs
- runtime state

Context engineering is the discipline of deciding how those pieces compete for a finite context window.

---

## 2. Why It Matters

As systems move from single-turn prompting to long-running agents and multi-step workflows, the main engineering problem shifts.

The challenge is no longer only:

- writing a better system prompt

It becomes:

- preventing the context window from filling with low-signal tokens
- preserving the information most relevant to the current step
- controlling latency and token cost
- avoiding position-related failures such as "Lost in the Middle"
- keeping multi-turn state coherent over time

Large context windows help, but they do not remove the problem. More tokens increase opportunity, but they also increase noise, attention dilution, and cost. Context should therefore be treated as a **finite, high-value working memory**, not an infinite dump buffer.

---

## 3. Typical Use Cases

You usually need context engineering when the task depends on one or more of the following:

- multi-turn continuity
- external retrieval
- long documents
- tool use
- persistent user or task state
- multimodal inputs

For a dedicated breakdown of how CE strategy changes across Q&A, support, document analysis, coding, agents, personalization, and multimodal workflows, see [02_Context_Strategies_by_Scenario](./02_Context_Strategies_by_Scenario.md).

---

## 4. When You Probably Do Not Need Much CE

You may not need a heavy context engineering layer if your task is:

- single-turn and short
- mostly templated
- based on fixed instructions with little external context
- simple extraction, classification, or formatting

In those cases, disciplined prompt design is often enough. Context engineering becomes important once runtime state, retrieval, long history, or tool interaction starts to accumulate.

---

## 5. The Core Problems CE Solves

This module treats context engineering as a response to five recurring production problems:

### 5.1 Selection

There is always more potentially relevant information than fits in the window.

Question:

- what should be included for this call?

### 5.2 Ordering

Placement affects model behavior.

Question:

- what should go at the beginning, middle, and end?

### 5.3 Budgeting

Input and output share a finite token budget.

Question:

- how much of the budget should go to instructions, memory, RAG, history, tools, and output reserve?

### 5.4 Compression

Long-running sessions inevitably exceed budget.

Question:

- what should be summarized, truncated, converted to structured state, or dropped?

### 5.5 Observability

Without instrumentation, CE decisions remain guesswork.

Question:

- how do we measure utilization, compression quality, cost, and context effectiveness?

---

## 6. CE vs. Neighboring Topics

Context engineering overlaps with several nearby topics, but it is not identical to them.

| Topic | Primary Question | Covered In |
| :--- | :--- | :--- |
| Prompt Engineering | How should instructions be written? | [../../02_Prompt_Engineering](../../02_Prompt_Engineering) |
| Context Engineering | What information should be present during this inference step? | This module |
| Memory | How should information persist across sessions? | [../../04_Memory](../../04_Memory) |
| RAG | How should external information be retrieved? | [../../05_RAG](../../05_RAG) |
| Agents | How should tools, plans, and actions be orchestrated? | [../../06_Agent](../../06_Agent) |

One useful rule:

- **RAG retrieves**
- **Memory persists**
- **Prompting instructs**
- **Agents act**
- **Context engineering decides what actually enters the model's working set**

---

## 7. A Minimal Mental Model

For this repository, a good default mental model is:

1. **Load** candidate context from prompts, memory, retrieval, history, and tools.
2. **Budget** the available window and reserve output space.
3. **Compress or degrade** lower-priority content when needed.
4. **Assemble** the final context in an order that supports model attention.
5. **Observe** token use, cost, trimming decisions, and outcome quality.

If you remember only one thing, remember this:

> Context engineering is runtime information allocation under constraint.

### 7.1 A Practical Implementation Principle

In practice, most teams do **not** begin by building a perfect intent classifier, a full routing graph, or a complete archive activation system.

A more realistic engineering path is:

1. start with a simple default context strategy
2. observe where that default fails in evals or production
3. introduce retrieval, archive recall, routing, or structured memory only where repeated failures justify the added complexity

This matters because a "decision layer" is itself a system with cost, latency, and failure modes. If introduced too early, it can become harder to justify than the context problem it was meant to solve.

So a practical CE philosophy is usually:

- start with a business-acceptable default
- measure recurring failures
- add control layers incrementally

In other words:

> Do not start with maximum sophistication. Start with the simplest strategy that works, then upgrade only where reality forces you to.

---

## 8. What This Module Will Teach

This module is organized to move from definition to mechanics to production patterns.

- [02_Context_Strategies_by_Scenario](./02_Context_Strategies_by_Scenario.md): how CE strategy changes across tasks
- [03_Context_Window_Mechanics](./03_Context_Window_Mechanics.md): context window as constrained working memory
- [04_Context_Composition](./04_Context_Composition.md): context layers, ordering, and priority
- [05_Token_Budget_and_Cost](./05_Token_Budget_and_Cost.md): token allocation and compression policy
- [06_Long_Context_Techniques](./06_Long_Context_Techniques.md): chunking, map-reduce, tree-of-summaries
- [07_Dynamic_Context_Management](./07_Dynamic_Context_Management.md): multi-turn evolution, schema-driven state tracking
- [08_Advanced_Context_Paradigms](./08_Advanced_Context_Paradigms.md): advanced compression and orchestration ideas
- [09_CE_Evaluation](./09_CE_Evaluation.md): measuring whether CE decisions actually work

The practical layer then turns those ideas into shared primitives and case studies.

---

## 9. Recommended Next Step

After this introduction, continue to [02_Context_Strategies_by_Scenario](./02_Context_Strategies_by_Scenario.md), then [03_Context_Window_Mechanics](./03_Context_Window_Mechanics.md).

That document establishes the physical constraints behind all later CE decisions: finite budget, KV cache cost, prefix caching, position effects, and the four-step production pipeline.

---

## Reference

- Anthropic Engineering. [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents). Published September 29, 2025.
