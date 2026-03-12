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
| [01_Context_Window_Mechanics](01_Theory/01_Context_Window_Mechanics.md) | KV cache cost model, prefix caching, Lost in the Middle, effective vs nominal window | Prompt Engineering 01 |
| [02_Context_Composition](01_Theory/02_Context_Composition.md) | 7-layer context anatomy, Sandwich Pattern, priority hierarchy, multi-modal costs | Theory 01 |
| [03_Token_Budget_and_Cost](01_Theory/03_Token_Budget_and_Cost.md) | Budget model, priority-based allocation, compression strategies, cost optimization | Theory 02 |
| [04_Long_Context_Techniques](01_Theory/04_Long_Context_Techniques.md) | NIAH testing, chunking strategies, map-reduce, tree of summaries, position-aware placement | Theory 02 |
| [05_Dynamic_Context_Management](01_Theory/05_Dynamic_Context_Management.md) | Context as state machine, compression triggers, agent scratchpad, multi-agent isolation | Theory 03 + 04 |

---

## 02 Practical

| File | What It Implements | Prerequisite |
| :--- | :--- | :--- |
| [01_Context_Composition_Pipeline.py](02_Practical/01_Context_Composition_Pipeline.py) | `ContextLayer`, `ContextComposer`, priority-based trimming, Sandwich Pattern | Theory 02, 03 |
| [02_Token_Budget_Controller.py](02_Practical/02_Token_Budget_Controller.py) | `TokenBudgetController`, per-layer allocation, compression triggers | Theory 03 + Practical 01 |
| [03_Context_Compression.py](02_Practical/03_Context_Compression.py) | 5 compression strategies: truncation, sliding window, extractive, abstractive, entity | Theory 03 + Practical 02 |
| [04_Long_Document_Processor.ipynb](02_Practical/04_Long_Document_Processor.ipynb) | Chunking strategies, map-reduce, hierarchical summarization, position-aware assembly | Theory 04 |
| [05_Multi_Turn_Context_Manager.py](02_Practical/05_Multi_Turn_Context_Manager.py) | `MultiTurnContextManager`, compression triggers, checkpointing, session export | Theory 05 + Practical 01–03 |
| [06_Context_Observability.py](02_Practical/06_Context_Observability.py) | Token usage tracking, cost attribution, context diff logging, OpenTelemetry spans | Practical 01–02 |

---

## 03 Best Practice

| File | Topics |
| :--- | :--- |
| [01_Context_Architecture_Patterns](03_Best_Practice/01_Context_Architecture_Patterns.md) | 4 architecture patterns, decision tree, Static-First Rule, anti-patterns, production checklist |
| [02_Context_Quality_and_Evaluation](03_Best_Practice/02_Context_Quality_and_Evaluation.md) | NIAH, context relevance score, utilization rate, ROUGE-L, A/B testing |
| [03_Production_Context_Optimization](03_Best_Practice/03_Production_Context_Optimization.md) | Async assembly, prefix cache ROI, tiered models, graceful degradation, observability |

---

## Recommended Learning Path

```
Theory 01 → Theory 02 → Theory 03 ──→ Practical 01 → Practical 02 → Practical 03
                │                                                          │
                └──→ Theory 04 ──→ Practical 04                           │
                │                                                          ▼
                └──→ Theory 05 ──→ Practical 05 ──────────────→ Practical 06
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
