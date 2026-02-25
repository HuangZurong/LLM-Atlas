# Model-Specific Quirks & Industrial Optimizations

*Prerequisite: [../01_Theory/01_Foundations_and_Anatomy.md](../01_Theory/01_Foundations_and_Anatomy.md).*

---

In 2025, each major LLM family has subtle but important differences in how they respond to prompts. An industrial engineer knows these details and adapts accordingly.

## 1. Prompt Format Preference Matrix

| Model Family | Preferred Structure | Common Pitfalls | Industrial Tip |
| :--- | :--- | :--- | :--- |
| **GPT (OpenAI)** | JSON with `json_mode`, Markdown bullets | Can be "lazy" (short answers) | Use explicit length constraints: "Output a 200-word summary." |
| **Claude (Anthropic)** | XML tags (`<answer>...</answer>`), Strong Role descriptions | Too verbose by default | Use "Be concise." and specify word limits. |
| **Llama (Meta)** | Standard chat format, Few-shot with reasoning examples | Prone to hallucination on unfamiliar topics | Provide strong context anchors. |
| **DeepSeek (R1)** | Reasoning models; prefer high-level goals over step-by-step. | Can over-think (high reasoning token cost) | Use strict constraints and allow large `max_completion_tokens`. |
| **o1 (OpenAI)** | Less is more. Avoid CoT prompts. | Slow response time due to internal reasoning. | Use for high-stakes logic, not simple tasks. |

## 2. Tokenization Bias & Impact

- **GPT (cl100k_base)**: Extra spaces cost tokens. `' cat'` and `'cat'` are different token IDs. Be strict with whitespace.
- **Claude**: Better multilingual tokenization. Chinese may be more efficient.
- **Llama**: Uses SentencePiece BPE; more forgiving of typos.
- **DeepSeek (R1)**: "Reasoning tokens" are billed but hidden; can surprise you on cost.

## 3. Guardrail & Safety Behavior

- **GPT**: Has a strong "Moderation API" layer; model itself is balanced.
- **Claude**: Strong "Refusal" behavior. You must explicitly allow "dangerous" tasks with a safety break-glass clause.
- **Llama (Open-weight)**: No built-in safety. You MUST implement external guardrails (as in `01_LLMs/02_Practical/08_Safety_Guardrails_Middleware.py`).

## 4. Performance & Latency Profile

| Model | Typical Latency (p50) | Key Use Case | Industrial Take |
| :--- | :--- | :--- | :--- |
| GPT-4o-mini | ~300ms | High-throughput classification, summarization | Batch API for massive offline workloads. |
| GPT-4o | ~700ms | General reasoning, complex extraction | Use with caching and failover to mini for cost control. |
| Claude (Sonnet) | ~1.2s | Long-document analysis, legal/medical | Great for "read-then-act" tasks. |
| DeepSeek-R1 | ~5-30s | Math, Code, Puzzles | Not real-time. Ideal for offline verification loops. |
| Llama 3.1 (70B) | ~4s (self-hosted) | On-premise data security | Use quantized models for real-time, but watch quality degradation. |

## 5. Summary: When to Use Which

**For general chatbots**: GPT-4o-mini (cost) + GPT-4o (quality fallback) + Claude (safety fallback).
**For agentic systems**: GPT-4o (primary) + DeepSeek-R1 (reasoning fallback).
**For high-security on-prem**: Llama 3.1 + local guardrails.