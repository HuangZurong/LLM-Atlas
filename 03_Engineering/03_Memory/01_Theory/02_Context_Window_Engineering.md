# Context Window Engineering

*Prerequisite: [01_Memory_Systems.md](01_Memory_Systems.md).*

---

The context window is the most expensive and constrained resource in LLM applications. This guide covers the low-level mechanics and industrial strategies for maximizing its utility.

## 1. The KV Cache: Why Context is Expensive

Every token in the context window generates a **Key-Value (KV) pair** in the Transformer's attention layers. These pairs are stored in GPU memory (VRAM).

### Cost Model:
```
KV Cache Memory = 2 × num_layers × hidden_dim × num_tokens × precision_bytes
```
- For a 70B model with 80 layers, 8192 hidden dim, FP16: ~640 bytes per token per layer.
- A 128K context window can consume **10+ GB of VRAM** per request.

### Industrial Implication:
Every token you put into the context window has a **triple cost**:
1. **Latency**: More tokens = slower Time-to-First-Token (TTFT).
2. **Memory**: More KV entries = fewer concurrent requests on the same GPU.
3. **Billing**: Input tokens are billed (often at 50% of output token rate).

## 2. Prefix Caching (KV Cache Reuse)

When multiple requests share the same prefix (System Prompt + Few-shot examples), the KV Cache for that prefix can be **reused** across requests.

### How it works:
```
Request 1: [System Prompt][Examples][User Query A]
                ↑ cached ↑
Request 2: [System Prompt][Examples][User Query B]
                ↑ reused ↑
```

### Provider Support (2025):
| Provider | Mechanism | Discount |
| :--- | :--- | :--- |
| **OpenAI** | Automatic Prefix Caching | ~50% on cached input tokens |
| **Anthropic** | `cache_control` parameter | ~90% on cached tokens |
| **DeepSeek** | Automatic (>32 tokens prefix) | ~90% on cached tokens |
| **vLLM (Self-hosted)** | `--enable-prefix-caching` flag | Free (you own the GPU) |

### Engineering Rule:
**Static content FIRST, dynamic content LAST.** This maximizes the cacheable prefix length.

## 3. Attention Patterns & "Lost in the Middle"

Research (Liu et al., 2023) shows that LLMs attend most strongly to:
1. The **beginning** of the context (primacy bias).
2. The **end** of the context (recency bias).
3. Information in the **middle** is often ignored.

### Industrial Mitigation:
- **Sandwich Pattern**: Place critical instructions at both the start AND end.
- **Chunked Retrieval**: When injecting RAG results, place the most relevant chunk at the END, not in the middle.
- **NIAH Testing**: Use "Needle in a Haystack" tests to validate your model's retrieval accuracy at different context positions.

## 4. Context Window Sizing Strategy

Not all tasks need 128K tokens. Over-provisioning wastes money and increases latency.

| Task Type | Recommended Context Budget | Reasoning |
| :--- | :--- | :--- |
| Simple Q&A | 4K - 8K | System prompt + short query |
| Multi-turn Chat | 16K - 32K | History + current turn |
| Document Analysis | 32K - 128K | Full document + instructions |
| Code Generation | 8K - 32K | Relevant files + task description |
| Agent with Tools | 16K - 64K | History + tool outputs accumulate fast |

## 5. Dynamic Context Composition

In production, the context is **assembled dynamically** from multiple sources:

```
┌─────────────────────────────────────────────┐
│ System Prompt (Fixed, Cached)         ~500T │
│ Retrieved Memory (Vector DB)        ~2000T  │
│ RAG Context (Retrieved Docs)        ~4000T  │
│ Conversation History (Last N turns) ~3000T  │
│ Current User Query                   ~200T  │
│ ─────────────────────────────────────────── │
│ TOTAL                               ~9700T  │
│ Budget Remaining (for output)       ~6300T  │
└─────────────────────────────────────────────┘
```

The **Memory Manager** is responsible for allocating tokens across these sources while staying within budget.
