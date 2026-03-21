# 03 · Context Engineering

*Prerequisite: [02_Prompt_Engineering](../02_Prompt_Engineering) — understand how to write effective prompts before managing what goes into the context window.*

---

Context engineering is the discipline of deciding **what to put in the context window, in what order, and at what priority**. While prompt engineering focuses on *how* to phrase instructions, context engineering focuses on the *composition and lifecycle* of the entire context.

In production, the context is assembled dynamically from multiple sources — system prompt, retrieved memory, RAG results, conversation history, tool outputs — all competing for a finite token budget. Getting this right is the difference between a reliable application and one that hallucinates, truncates, or costs 10× more than necessary.

---

## Module Structure

```
03_Context_Engineering/
├── 01_Theory/          Concepts and mental models
├── 02_Practical/       Working implementations
└── 03_Best_Practice/   Production patterns and decision frameworks
```

---

## 01 Theory

| File | Topics | Prerequisite |
| :--- | :--- | :--- |
| [01_Context_Window_Mechanics](01_Theory/01_Context_Window_Mechanics.md) | **[Foundation]** The "One Core, Two Axes" mental model + KV cache cost, prefix caching, Lost in the Middle | Prompt Engineering 01 |
| [02_Context_Composition](01_Theory/02_Context_Composition.md) | 7-layer context anatomy, Sandwich Pattern, priority hierarchy, multi-modal costs | Theory 01 |
| [03_Token_Budget_and_Cost](01_Theory/03_Token_Budget_and_Cost.md) | Budget model, priority-based allocation, compression strategies, cost optimization | Theory 02 |
| [04_Long_Context_Techniques](01_Theory/04_Long_Context_Techniques.md) | NIAH testing, chunking strategies, map-reduce, tree of summaries, position-aware placement | Theory 02 |
| [05_Dynamic_Context_Management](01_Theory/05_Dynamic_Context_Management.md) | Context as state machine, Schema-Driven State Tracking, hybrid schema, tiered model routing, memory consolidation | Theory 03 + 04 |
| [06_Advanced_Context_Paradigms](01_Theory/06_Advanced_Context_Paradigms.md) | Information-theoretic compression, structure/modality-aware budgeting, MAS context orchestration, advanced evaluation | Theory 03 + 05 |
| [07_CE_Evaluation](01_Theory/07_CE_Evaluation.md) | Evaluation metrics, NIAH variants, context relevance scoring, benchmarking methodology | Theory 04 + 06 |

---

## 02 Practical

### Shared Library

Reusable primitives imported by all case studies.

| File | What It Implements | Prerequisite |
| :--- | :--- | :--- |
| [shared/composer.py](02_Practical/shared/composer.py) | `ContextLayer`, `ContextComposer`, priority-based trimming, Sandwich Pattern | Theory 02, 03 |
| [shared/budget_controller.py](02_Practical/shared/budget_controller.py) | `TokenBudgetController`, per-layer allocation, compression triggers | Theory 03 |
| [shared/compressor.py](02_Practical/shared/compressor.py) | 5 compression strategies: truncation, sliding window, extractive, abstractive, entity | Theory 03 |
| [shared/observability.py](02_Practical/shared/observability.py) | `ContextObserver`, cost attribution, context diff logging, OpenTelemetry spans | Theory 01–03 |

### Case Studies

End-to-end scenarios combining the shared library to solve real problems.

| Directory | What It Demonstrates | Prerequisite |
| :--- | :--- | :--- |
| [customer_support/](02_Practical/customer_support/) | Multi-turn context management, schema-driven state tracking, compression triggers | Theory 05 + shared/ |
| [document_analysis/](02_Practical/document_analysis/) | Chunking strategies, map-reduce, hierarchical summarization, position-aware assembly | Theory 04 + shared/ |

---

## 03 Best Practice

| File | Topics |
| :--- | :--- |
| [01_Context_Architecture_Patterns](03_Best_Practice/01_Context_Architecture_Patterns.md) | 4 architecture patterns, decision tree, Static-First Rule, anti-patterns, production checklist |
| [02_Context_Quality_and_Evaluation](03_Best_Practice/02_Context_Quality_and_Evaluation.md) | NIAH, context relevance score, utilization rate, ROUGE-L, A/B testing |
| [03_Production_Context_Optimization](03_Best_Practice/03_Production_Context_Optimization.md) | Async assembly, prefix cache ROI, tiered models, graceful degradation, observability |
| [04_Anthropic_CE_Practices](03_Best_Practice/04_Anthropic_CE_Practices.md) | Smallest high-signal token set, Context Rot, Compaction, JIT loading |
| [05_OpenAI_CE_Practices](03_Best_Practice/05_OpenAI_CE_Practices.md) | 4-phase memory pipeline, Compaction API, RunContextWrapper |
| [06_Google_CE_Practices](03_Best_Practice/06_Google_CE_Practices.md) | State as Context Bus, include_contents="none", delta tracking |
| [07_Cursor_CE_Practices](03_Best_Practice/07_Cursor_CE_Practices.md) | Tiered retrieval, embedding recall + reranking, speculative edits |
| [08_Frameworks_CE_Practices](03_Best_Practice/08_Frameworks_CE_Practices.md) | LangGraph reducers, CrewAI entity memory, MemGPT self-managed memory |
| [09_CE_Cross_Comparison](03_Best_Practice/09_CE_Cross_Comparison.md) | Philosophy, composition, and performance data comparison |
| [10_CE_Framework_Design](03_Best_Practice/10_CE_Framework_Design.md) | Design doc for a highly configurable, vendor-agnostic CE framework |

---

## Recommended Learning Path

```
Theory 01 → Theory 02 → Theory 03 ──→ shared/composer
                │                   → shared/budget_controller
                │                   → shared/compressor
                │                   → shared/observability
                │                          │
                └──→ Theory 04 ────────────┼──→ document_analysis/
                │                          │
                └──→ Theory 05 ────────────┴──→ customer_support/
                │
                └──→ Theory 06 (Academic Advanced)
                                                    │
                                                    ▼
                                       Best Practice 01 → 02 → 03
```

---

## Scope Boundaries

| This module covers | Covered elsewhere |
| :--- | :--- |
| What goes in the context window | How to write instructions → [02_Prompt_Engineering](../02_Prompt_Engineering) |
| Token budget and compression | Cross-session persistence → [04_Memory](../04_Memory) |
| Long context processing (in-window) | External retrieval → [05_RAG](../05_RAG) |
| Agent context management | Agent architecture and tool use → [06_Agent](../06_Agent) |
