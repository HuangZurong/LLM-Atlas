# LLM Engineering Best Practices

*Prerequisite: [04_Context_Optimization.md](04_Context_Optimization.md).*

Building production systems with LLMs requires shifting from "prompt-and-pray" to rigorous engineering. This document covers the core patterns for stability, performance, and cost.

---

## 1. Reliability & Resilience Patterns

### 1.1 The Semantic Fallback
Don't just retry on 500 errors. Implement a tiered fallback:
1. **Primary**: High-intelligence model (e.g., GPT-4o)
2. **Fallback**: Faster, high-reliability model (e.g., Claude 3.5 Sonnet)
3. **Emergency**: Local small model or static "fail-soft" response.

### 1.2 Output Validation (The "Parser-Guard" Pattern)
Never trust LLM output. Always:
- Use **JSON Schema** for structured tasks.
- Use a **Heuristic Parser** (regex/Pydantic) to validate keys.
- **Auto-Retry on Error**: Feed the error message back to the LLM once to let it self-correct.

## 2. Performance Engineering

### 2.1 Prompt Caching (Prefix Optimization)
Most latency comes from the **Prefill** phase (processing your 2000-token system prompt).
- **Practice**: Keep the system prompt *identical* across requests.
- **Practice**: Place changing variables (user input) at the *end* of the prompt.
- **Impact**: 2x-10x faster Time-To-First-Token (TTFT) via KV cache reuse.

### 2.2 Speculative Decoding
For self-hosted models, use a small draft model (e.g., Qwen-1.5B) to predict tokens, which the main model (e.g., DeepSeek-V3) verifies in parallel.
- **Impact**: 1.5x-2x increase in throughput for sequential generation.

## 3. Cost Engineering

### 3.1 The "Intelligence-Density" Tradeoff
Calculate **Cost per Accurate Response**, not Cost per Token.
- Sometimes GPT-4o is cheaper than GPT-4o-mini because it solves the task in 1 attempt vs. 3 attempts + complex retry logic for the mini model.

### 3.2 Token Pruning
- Use **Long-Context Summarization** to compress RAG context.
- Strip system prompts to the bare essentials once a task is stable.
- Remove redundant whitespace and stop words (carefully).

## 4. Operational Best Practices

### 4.1 Prompt Versioning
Treat prompts as **deployment artifacts**.
- Never hardcode prompts in application code.
- Store them in a registry with versions: `summary_v1`, `summary_v1.1_cot`.
- Bind specific prompts to specific model versions.

### 4.2 Logging & Tracing (Golden Rules)
1. **Log everything**: Prompt (including variables), Full Response, Token Usage, Latency.
2. **Trace the Chain**: If using RAG, log which chunks were retrieved for which query.
3. **User Feedback loop**: Capture thumbs-up/down in the same trace to build your own fine-tuning/eval dataset.

## 5. Summary Matrix: Scientist vs. Engineer

| Topic | Scientist View | Engineer View |
|---|---|---|
| **Objective** | Accuracy / Benchmark score | Cost / Latency / Reliability |
| **Model** | Biggest model possible | Smallest model that does the job |
| **Prompt** | Creative / Sophisticated | Deterministic / Short / Cachable |
| **Output** | Natural Language | Structured JSON / Machine-readable |

