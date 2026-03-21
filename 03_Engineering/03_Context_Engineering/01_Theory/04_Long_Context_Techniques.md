# Long Context Techniques

*Prerequisite: [02_Context_Composition.md](02_Context_Composition.md).*

---

Long context handling addresses the challenge of processing documents, codebases, or conversation histories that approach or exceed the model's effective context window. This is distinct from RAG (which handles retrieval from external stores) — here we focus on what happens **inside** the context window.

## 1. The Long Context Problem Space

```
Document Size vs. Strategy:

< 8K tokens    → Fit entirely in context, no special handling needed
8K – 32K       → Selective inclusion: prioritize relevant sections
32K – 128K     → Chunking + hierarchical processing
> 128K         → RAG (see [05_RAG module](../../05_RAG/01_Theory/01_Architecture.md)) or multi-pass processing
```

The challenge is not just fitting tokens — it's maintaining **reasoning quality** across long spans. Models degrade in two ways:
1. **Retrieval degradation**: Failing to locate specific facts (Lost in the Middle).
2. **Reasoning degradation**: Losing coherence when reasoning chains span many tokens.

## 2. Needle-in-a-Haystack (NIAH) Testing

NIAH is the standard benchmark for evaluating long-context reliability. A single "needle" fact is inserted at a specific position in a long "haystack" document, and the model is asked to retrieve it.

```
Haystack: [Filler text ... ] [NEEDLE: "The secret code is 7392"] [... Filler text]
Query: "What is the secret code?"
```

### What NIAH Reveals

- **Position sensitivity**: Most models perform well at 0% and 100% depth, poorly at 40–60%.
- **Context length cliff**: Performance often drops sharply beyond 60–70% of nominal window.
- **Model-specific patterns**: Different models have different "blind spots."

### Running NIAH Before Deployment

Always run NIAH tests on your specific model + context length combination before deploying long-context features. Don't rely on published benchmarks alone — they use different haystack content than your domain.

**Advanced NIAH: Multi-Needle Reasoning**
Standard NIAH only tests if a model can *find* a single string. In production, models often need to synthesize multiple facts. Advanced evaluations (as discussed in `06_Advanced_Context_Paradigms.md`) insert 3-5 related needles across the 100K window to test if the model can connect them logically, which is a much stricter test of true long-context capability.

## 3. Chunking Strategies

When a document must be split, the chunking strategy determines what information is preserved and what is lost at boundaries.

### Fixed-Size Chunking

```
[Token 0 – 512] [Token 513 – 1024] [Token 1025 – 1536] ...
```
- Simple, predictable.
- Breaks sentences and paragraphs arbitrarily.
- Use only when document structure is irrelevant.

### Semantic Chunking

Split at natural boundaries: paragraphs, sections, sentences. Preserve structural units.

```python
# Split at double newlines (paragraph boundaries)
chunks = text.split("\n\n")
# Then merge small chunks and split large ones to target size
```

### Hierarchical Chunking

Maintain a two-level representation:
- **Summary level**: One summary per section (~100T each).
- **Detail level**: Full text of each section (~500T each).

Use the summary level for initial retrieval, then load the full detail level for the relevant section only.

### Sliding Window with Overlap

```
Window 1: [Token 0 – 512]
Window 2: [Token 384 – 896]   ← 128-token overlap
Window 3: [Token 768 – 1280]  ← 128-token overlap
```

Overlap prevents information loss at boundaries. Use 10–25% overlap.

## 4. Multi-Pass Processing

For documents too long to process in a single pass, use iterative strategies:

### Map-Reduce

```
Document
    │
    ├── Chunk 1 → [LLM: extract key facts] → Summary 1
    ├── Chunk 2 → [LLM: extract key facts] → Summary 2
    ├── Chunk 3 → [LLM: extract key facts] → Summary 3
    │
    └── [LLM: synthesize summaries] → Final Answer
```

Best for: summarization, information extraction, Q&A over long documents.

### Refine (Sequential)

```
Chunk 1 → [LLM] → Running Summary v1
Chunk 2 + Summary v1 → [LLM] → Running Summary v2
Chunk 3 + Summary v2 → [LLM] → Running Summary v3
...
Final Summary vN → Answer
```

Best for: tasks requiring coherent narrative (e.g., document summarization where order matters).

### Tree of Summaries

```
Level 0 (raw):    [C1][C2][C3][C4][C5][C6][C7][C8]
Level 1 (pairs):  [S12]    [S34]    [S56]    [S78]
Level 2 (quads):  [S1234]           [S5678]
Level 3 (root):   [S12345678]
```

Best for: very long documents where a single map-reduce pass loses too much detail.

## 5. Position-Aware Context Placement

Given attention patterns, place content strategically:

| Content Type | Optimal Position | Reason |
| :--- | :--- | :--- |
| Task instructions | START + END | Primacy + recency bias |
| Most relevant chunk | END (just before query) | Recency bias |
| Background context | START (after system prompt) | Primacy bias |
| Less relevant chunks | MIDDLE | Acceptable loss zone |
| Output format spec | END | Model acts on last instructions |

## 6. Long Context in Agent Loops

*Note: For the definitive patterns on managing long contexts in autonomous agents, refer to `05_Dynamic_Context_Management.md` and `06_Advanced_Context_Paradigms.md`.*

In multi-step agent execution, context grows with each tool call. Manage this proactively:

```
Turn 1:  [System][Task][Tool Schema]                    → 2K tokens
Turn 5:  [System][Task][Tool Schema][4x Tool Results]   → 8K tokens
Turn 15: [System][Task][Tool Schema][14x Tool Results]  → 25K tokens
Turn 30: [System][Task][Tool Schema][29x Tool Results]  → 50K tokens  ← danger zone
```

**Strategies**:
- **Result pruning**: After each tool call, discard results that are no longer needed.
- **Scratchpad summarization**: Every N turns, summarize the reasoning trace into a compact state.
- **Context checkpointing**: Save the current state to external memory, reset context, inject the checkpoint summary.

---

## Key References

1. **Liu, N. F., et al. (2024). Lost in the Middle.** *TACL, 12*, 157–173.
2. **Hsieh, C., et al. (2024). RULER: What's the Real Context Size of Your Long-Context Language Models?** *arXiv:2404.06654*.
3. **Anthropic. (2025). Effective Context Engineering for AI Agents.** Anthropic Engineering Blog.
4. **LangChain. (2024). Document Transformers — Text Splitters.** LangChain Documentation.
