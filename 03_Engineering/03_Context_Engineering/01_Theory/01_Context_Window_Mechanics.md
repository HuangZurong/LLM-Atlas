# 01 · Context Window Mechanics

*Prerequisite: [../../02_Prompt_Engineering/01_Theory/01_Foundations_and_Anatomy.md](../../02_Prompt_Engineering/01_Theory/01_Foundations_and_Anatomy.md).*

---

## The Context Engineering Mental Model

Before diving into mechanics, understand the overarching framework:

**One Core**: Treat the Context Window as **extremely expensive, highly constrained, and unreliable system RAM**. Context Engineering decides *what* to load, in *what order*, and at *what priority* when multiple sources compete for this limited space.

**Two Axes**:
- **Spatial Axis**: How to arrange information within a single request (Prefix Caching, Lost in the Middle, Sandwich Pattern).
- **Temporal Axis**: How context evolves across multi-turn conversations and agent loops (covered in `05_Dynamic_Context_Management.md`).

**Three Stages** of evolution:
1. Static Assembly → 2. Dynamic Budgeting → 3. Agentic Orchestration.

**Four-Step Pipeline** for production systems:
1. **Load** → 2. **Budget & Sort** → 3. **Compress & Degrade** → 4. **Assemble & Observe**.

*This document covers the Spatial Axis and the physical constraints that drive all budgeting decisions.*

---

## 1. What the Context Window Actually Is

The context window is the **total token budget** for a single LLM call — it includes both input (prompt) and output (completion). Everything the model "knows" during a call must fit inside it.

```
┌─────────────────────────────────────────────────────┐
│                  Context Window (e.g. 128K tokens)  │
│                                                     │
│  ← Input Tokens ──────────────────→ ← Output →     │
│  [System][Memory][RAG][History][Query]  [Response]  │
└─────────────────────────────────────────────────────┘
```

Unlike human memory, the context window is:
- **Flat**: No inherent hierarchy or importance weighting.
- **Ephemeral**: Discarded after the call ends.
- **Expensive**: Every token has a latency, memory, and billing cost.

## 2. The KV Cache: Why Context is Expensive

Every token in the context generates a **Key-Value (KV) pair** in each Transformer attention layer. These pairs are stored in GPU VRAM.

### Cost Model

```text
KV Cache Memory = 2 × num_layers × hidden_dim × num_tokens × precision_bytes
```

For a 70B model (e.g., Llama 3 70B: 80 layers, 8192 hidden dim, FP16):
- ~2.5 MB VRAM per 1,000 tokens
- A 128K context window consumes **~320 MB per request**.
- If serving 100 concurrent users at full context, KV cache alone requires **~32 GB of VRAM**.

This hardware constraint directly translates to cost limits on API calls.

### Triple Cost Per Token

| Cost Dimension | Impact |
| :--- | :--- |
| **Latency** | More tokens → slower Time-to-First-Token (TTFT) |
| **Memory** | More KV entries → fewer concurrent requests per GPU |
| **Billing** | Input tokens are billed (typically at 50% of output rate) |

### PagedAttention

vLLM's PagedAttention manages KV cache like an OS manages virtual memory — allocating non-contiguous physical memory blocks to logical sequences. This enables higher batch sizes and better GPU utilization without changing model behavior.

## 3. Prefix Caching (KV Cache Reuse)

When multiple requests share the same prefix (system prompt + few-shot examples), the KV cache for that prefix can be **reused** across requests.

```
Request 1: [System Prompt][Examples][User Query A]
                ↑ computed and cached ↑
Request 2: [System Prompt][Examples][User Query B]
                ↑ reused — zero recomputation ↑
```

### Provider Support (as of early 2025 — verify against official docs)

| Provider | Mechanism | Discount |
| :--- | :--- | :--- |
| **OpenAI** | Automatic Prefix Caching | ~50% on cached input tokens |
| **Anthropic** | `cache_control` parameter | ~90% on cached tokens |
| **DeepSeek** | Automatic (>32 token prefix) | ~90% on cached tokens |
| **vLLM (self-hosted)** | `--enable-prefix-caching` flag | Free (you own the GPU) |

### 3.1 The Prefix Caching Anti-Pattern: Dynamic Injections

A common pitfall that destroys caching efficiency is injecting dynamic variables (timestamps, UUIDs, or session IDs) into otherwise static blocks.

```text
❌ BAD: Breaks cache every second
System: "You are a helpful assistant. Current time: 2025-03-20 10:00:01"
User:   "What is my task?"

✅ GOOD: Preserves cache
System: "You are a helpful assistant."
User:   "[Time: 2025-03-20 10:00:01] What is my task?"
```

**Engineering Rule**: Isolate dynamic variables to the very end of the prompt (the user message layer). Never pollute the static `[System Prompt]` or `[Few-shot Examples]` with per-request data.

```
✅ [System Prompt][Few-shot Examples][Retrieved Docs][User Query]
   ←────── stable, cacheable ──────→ ←── dynamic ──→

❌ [User Query][System Prompt][Few-shot Examples][Retrieved Docs]
   ← dynamic → ←────────── cache miss every time ──────────────→
```

## 4. Attention Patterns & "Lost in the Middle"

Research (Liu et al., 2023) shows LLMs attend most strongly to:
1. The **beginning** of the context (primacy bias).
2. The **end** of the context (recency bias).
3. Information in the **middle** is frequently ignored or hallucinated.

```
Attention
Strength
  ▲
  │█                                                    █
  │██                                                  ██
  │███                                              ████
  │████                                          ██████
  │█████████                            ████████████████
  └──────────────────────────────────────────────────────→
  Start                                                End
                    Context Position
```

### Mitigations

- **Sandwich Pattern**: Place critical instructions at both the START and END of the prompt.
- **Recency Placement**: Put the most relevant RAG chunk at the END, not buried in the middle.
- **NIAH Testing**: Use Needle-in-a-Haystack tests to validate your model's retrieval accuracy at different context positions before deploying.

## 5. Effective Window vs. Nominal Window

Just because a model *supports* 128K tokens doesn't mean it can *reason* over 128K tokens effectively.

| Window Type | Definition |
| :--- | :--- |
| **Nominal Window** | The maximum tokens the model accepts without error |
| **Effective Window** | The range over which the model reliably retrieves and reasons |

Most models show measurable performance degradation well before hitting their nominal limit. The effective window is typically 60–80% of the nominal window for complex reasoning tasks.

**Practical implication**: Don't fill the context to 95% capacity. The model needs headroom to "breathe".

### 5.1 The Output Safety Margin

A critical oversight in naive implementations is filling the context to the brim with inputs, leaving no room for the generated response.
If the context budget is 128K and your input is 127.9K, the model will output 100 tokens and abruptly halt (`finish_reason="length"`).

**Engineering Rule**: Always define a strict `Output Reserve` (e.g., 4,000 tokens) that cannot be encroached upon by the input context composer.

## 6. Model Context Windows at a Glance (early 2025)

A reference for planning context budgets. Verify against official documentation — these change frequently.

| Model | Nominal Window | Notes |
| :--- | :--- | :--- |
| **Claude 3.5 Sonnet** | 200K tokens | Prefix caching via `cache_control` |
| **Claude 3.5 Haiku** | 200K tokens | Prefix caching via `cache_control` |
| **GPT-4o** | 128K tokens | Automatic prefix caching |
| **GPT-4o mini** | 128K tokens | Automatic prefix caching |
| **Gemini 1.5 Pro** | 1M tokens | Implicit caching |
| **Gemini 2.0 Flash** | 1M tokens | Implicit caching |
| **DeepSeek-V3** | 128K tokens | Automatic prefix caching (>32T) |
| **Llama 3.3 70B** | 128K tokens | Prefix caching via vLLM flag |
| **Qwen2.5 72B** | 128K tokens | Prefix caching via vLLM flag |

**Key insight**: a larger nominal window does not automatically mean better long-context reasoning. Always run NIAH tests at your target depth before relying on the full window.

---

## Key References

1. **Liu, N. F., et al. (2024). Lost in the Middle: How Language Models Use Long Contexts.** *TACL, 12*, 157–173.
2. **Kwon, W., et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention.** *SOSP 2023*, 611–626.
3. **Bertsch, A., et al. (2024). Needle In A Haystack: Evaluating Long-Context Language Models.** *arXiv:2407.05831*.
