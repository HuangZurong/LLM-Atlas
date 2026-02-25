# Context Engineering

*Prerequisite: [03_API_Mechanics.md](03_API_Mechanics.md).*

The Context Window is the "Short-term memory" of the LLM. It is its most precious and limited resource.

---

## 1. The Context Constraint

LLMs have a fixed maximum context window (e.g., 128k tokens). This includes **both** the input (prompt) and the generated output.

### 1.1 The KV Cache (Key-Value Cache)
During generation, the model caches the internal representations of all previous tokens so it doesn't have to recompute them.
- **Problem**: The KV cache grows linearly with sequence length.
- **Problem**: KV cache consumption takes up significant VRAM, limiting the batch size.
- **Solution**: **PagedAttention** (vLLM) manages KV cache like an OS manages virtual memory.

## 2. Managing Long Contexts

When your data exceeds the context window, you must apply engineering strategies:

| Strategy | Logic | Pros/Cons |
|---|---|---|
| **Truncation** | Drop the oldest tokens to stay within limit | Simple; but loses "beginning of conversation" context |
| **Summarization** | Summarize old turns into a few tokens | Preserves key info; lossy, expensive to call summarizer |
| **Sliding Window** | Only look at the most recent $N$ tokens | predictable; loses all long-range context |
| **RAG (Retrieval)** | Store long docs in vector DB, retrieve only relevant chunks | Scales to millions of tokens; retrieval might miss info |

## 3. Position Bias (The "Lost in the Middle" Phenomenon)

Research shows that models are best at recalling information placed at the **very beginning** or **very end** of a long context. Information in the middle is frequently ignored or hallucinated.

**Engineering Insight**: Place the most critical instructions and the most relevant context chunks at the ends of your prompt.

## 4. Context Window vs. Context "Effective" Window

Just because a model *supports* 128k tokens doesn't mean it can *reason* over 128k tokens.
- **Needle In A Haystack (NIAH)**: A test where a single fact is buried in a long context.
- **Result**: Most models show significant performance degradation well before they hit their physical limit.

## 5. Caching Prefixes (KV Cache Reuse)

Many user queries share the same system prompt or few-shot examples.
- **Prompt Caching**: If the first $N$ tokens of two requests are identical, the engine can skip the computation for that prefix and reuse the cached KV states.
- **Efficiency**: Reduces latency (TTFT) and cost for repetitive workloads.


---

## Key References

1. **Liu, N. F., et al. (2024). Lost in the Middle: How Language Models Use Long Contexts.** *Transactions of the Association for Computational Linguistics, 12*, 157–173.
2. **Bertsch, A., et al. (2024). Needle In A Haystack: Evaluating Long-Context Language Models.** *arXiv preprint arXiv:2407.05831*.
3. **Kwon, W., et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention.** *Proceedings of the 29th Symposium on Operating Systems Principles*, 611–626.
4. **Tworkowski, S., et al. (2024). Focused Transformer: Contrastive Training for Context Scaling.** *Advances in Neural Information Processing Systems, 36*.
