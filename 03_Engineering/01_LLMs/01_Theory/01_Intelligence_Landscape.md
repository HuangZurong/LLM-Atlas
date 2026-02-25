# LLM Selection & Integration Guide

*Prerequisite: None (Engineering Track entry point).*
*See Also: [../../../04_Solutions/01_Technology_Selection.md](../../../04_Solutions/01_Technology_Selection.md) (solution-level model selection), [../../../02_Scientist/01_Architecture/07_Architecture_Paradigms.md](../../../02_Scientist/01_Architecture/07_Architecture_Paradigms.md) (deep architecture theory).*

As an LLM engineer, the first decision before building any application is: **which model, and how to access it?** This document covers the model landscape, selection criteria, and integration patterns.

---

## 1. Model Landscape (2024–2025)

### 1.1 Closed-Source (API-only)

| Provider | Model | Strengths | Context Window |
|---|---|---|---|
| OpenAI | GPT-4o, o1, o3 | General-purpose leader, strong reasoning (o-series) | 128K |
| Anthropic | Claude 3.5 Sonnet, Claude Opus 4 | Long-context, instruction following, safety | 200K |
| Google | Gemini 2.0 Flash/Pro | Multimodal, speed (Flash), long context (1M) | 1M |

### 1.2 Open-Weights (Self-hostable)

| Model | Parameters | Strengths | License |
|---|---|---|---|
| Llama 3.1 | 8B / 70B / 405B | General-purpose, strong community ecosystem | Llama 3.1 Community |
| DeepSeek-V3 | 671B (MoE) | Cost-efficient training, strong coding/math | MIT |
| DeepSeek-R1 | 671B (MoE) | Reasoning (RLVR-trained), open CoT | MIT |
| Qwen 2.5 | 0.5B–72B | Multilingual (Chinese/English), diverse sizes | Apache 2.0 |
| Mistral | 7B / 8x7B (Mixtral) | Efficient MoE, strong for size | Apache 2.0 |

### 1.3 Specialized Models

| Category | Models | Use Case |
|---|---|---|
| Code | DeepSeek-Coder-V2, CodeLlama, StarCoder2 | Code generation, completion, review |
| Embedding | BGE-M3, E5-Mistral, text-embedding-3 | RAG retrieval, semantic search |
| Reranking | BGE-Reranker, Cohere Rerank | Second-stage retrieval ranking |
| Vision | LLaVA, Qwen-VL, GPT-4o | Image understanding, OCR |

## 2. Model Selection Criteria

Choosing a model is a **multi-dimensional optimization** problem:

```
                    Quality
                      ▲
                      │
                      │   ● GPT-4o / Claude Opus
                      │
                      │        ● DeepSeek-V3
                      │
                      │   ● Llama-70B
                      │
                      │        ● Qwen-7B
          ────────────┼──────────────────► Cost (lower is better)
                      │
                      │   ● GPT-4o-mini / Flash
                      │
                      ▼
                   Latency
```

### 2.1 Decision Framework

| Factor | Question to Ask | Impact |
|---|---|---|
| **Task complexity** | Does it require multi-step reasoning or simple extraction? | Complex → larger model; Simple → smaller/faster |
| **Context length** | How much input context is needed? | Long docs → Gemini (1M) or Claude (200K) |
| **Latency requirement** | Real-time chat or batch processing? | Real-time → smaller model or speculative decoding |
| **Cost budget** | Per-query cost tolerance? | High volume → self-host open model; Low volume → API |
| **Data privacy** | Can data leave your infrastructure? | Sensitive → self-host; Non-sensitive → API |
| **Language** | Primary language of users? | Chinese → Qwen/DeepSeek; Multilingual → GPT-4o |
| **Licensing** | Commercial use? Fine-tuning needed? | Check license (Llama Community vs Apache 2.0 vs proprietary) |

### 2.2 Model Routing (Advanced)

In production, a single model rarely fits all queries. **Model routing** dispatches requests to different models based on complexity:

```
User Query → Complexity Classifier → Easy  → Small model (fast, cheap)
                                   → Hard  → Large model (slow, expensive)
                                   → Code  → Code-specialized model
```

This can reduce cost by 50–70% while maintaining quality on hard queries.

## 3. API Integration Patterns

### 3.1 The OpenAI-Compatible Standard

Most providers and inference engines now support the OpenAI chat completions format:

```python
from openai import OpenAI

# Works with OpenAI, vLLM, Ollama, LiteLLM, etc.
client = OpenAI(
    base_url="http://localhost:8000/v1",  # swap endpoint for any provider
    api_key="your-key"
)

response = client.chat.completions.create(
    model="deepseek-v3",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain PagedAttention."}
    ],
    stream=True
)
```

### 3.2 Streaming

For chat applications, streaming is essential for perceived responsiveness:

- **SSE (Server-Sent Events)**: Standard for HTTP streaming. Each chunk contains a delta token.
- **TTFT (Time to First Token)**: The most important latency metric for user experience.
- **TPS (Tokens per Second)**: Determines how fast the response "types out."

### 3.3 Resilience Patterns

| Pattern | Implementation |
|---|---|
| **Retry with backoff** | Exponential backoff on 429/500 errors |
| **Timeout** | Set per-request timeout (e.g., 30s for chat, 120s for long generation) |
| **Fallback** | Primary model fails → fallback to secondary model |
| **Circuit breaker** | After N consecutive failures, stop calling and return cached/default response |

### 3.4 Multi-Provider Abstraction

Libraries like **LiteLLM** provide a unified interface across 100+ providers:

```python
import litellm

# Same interface, different providers
response = litellm.completion(
    model="gpt-4o",           # or "claude-3-5-sonnet", "deepseek/deepseek-chat"
    messages=[{"role": "user", "content": "Hello"}]
)
```

## 4. Local vs Cloud Deployment

| Dimension | Cloud API | Self-Hosted |
|---|---|---|
| **Setup time** | Minutes | Days–weeks |
| **Cost at low volume** | Low (pay per token) | High (GPU idle cost) |
| **Cost at high volume** | High (token costs add up) | Low (amortized GPU cost) |
| **Data privacy** | Data leaves your infra | Data stays local |
| **Customization** | Limited (prompt only) | Full (fine-tune, modify) |
| **Maintenance** | Zero | You own it (updates, scaling, monitoring) |

**Rule of thumb**: Start with cloud API for prototyping. Move to self-hosted when you hit one of: (1) data privacy requirements, (2) >$10K/month API spend, (3) need for fine-tuning.

## 5. Inference Engines

| Engine | Best For | Key Feature |
|---|---|---|
| **vLLM** | Production serving | PagedAttention, continuous batching, highest throughput |
| **TGI** | HuggingFace ecosystem | Easy deployment, good defaults |
| **Ollama** | Local development | One-command setup, model library |
| **llama.cpp** | CPU/edge inference | GGUF quantization, runs on laptops |
| **SGLang** | Complex generation patterns | RadixAttention, structured generation |

### 5.1 Key Metrics

- **Throughput**: Total tokens/second across all concurrent requests
- **TTFT**: Time from request to first token (user-perceived latency)
- **TPS**: Tokens per second per request (generation speed)
- **Memory**: Peak VRAM usage (determines which GPU you need)

## 6. Token Economics

### 6.1 Tokenizer Differences

Different model families use different tokenizers. The same text produces different token counts:

| Text | GPT-4o (tiktoken) | Llama 3 (SentencePiece) | Qwen (custom) |
|---|---|---|---|
| "Hello, world!" | 3 tokens | 4 tokens | 3 tokens |
| Chinese text (100 chars) | ~100 tokens | ~150 tokens | ~80 tokens |

This directly impacts cost and context window utilization.

### 6.2 Cost Optimization Strategies

| Strategy | Savings | Tradeoff |
|---|---|---|
| **Prompt caching** | 50–90% on repeated prefixes | Only works with shared system prompts |
| **Shorter prompts** | Linear cost reduction | May reduce quality |
| **Model routing** | 50–70% overall | Adds routing complexity |
| **Batch API** | 50% (OpenAI Batch) | Higher latency (24h window) |
| **Quantized self-hosting** | 80%+ vs API at scale | Infra maintenance overhead |

## 7. Evaluation & Benchmarking

### 7.1 Standard Benchmarks

| Benchmark | What It Measures |
|---|---|
| **MMLU** | Broad knowledge across 57 subjects |
| **HumanEval / MBPP** | Code generation correctness |
| **MT-Bench** | Multi-turn conversation quality (LLM-as-Judge) |
| **Arena Elo** | Human preference ranking (Chatbot Arena) |
| **MATH / GSM8K** | Mathematical reasoning |
| **IFEval** | Instruction following precision |

### 7.2 Task-Specific Evaluation

Benchmarks are a starting point, not the answer. For your specific use case:

1. Build a **golden test set** of 50–200 representative queries with expected outputs
2. Define **task-specific metrics** (accuracy, format compliance, latency)
3. Run **A/B tests** in production with real users
4. Track **cost per successful query** (not just cost per token)

---

## Key References

1. **Brown, T. B., et al. (2020). Language Models are Few-Shot Learners.** *Advances in Neural Information Processing Systems, 33*, 1877–1901.
2. **Touvron, H., et al. (2023). LLaMA: Open and Efficient Foundation Language Models.** *arXiv preprint arXiv:2302.13971*.
3. **OpenAI (2023). GPT-4 Technical Report.** *arXiv preprint arXiv:2303.08774*.
4. **Anthropic (2024). Claude 3.5 Sonnet: Technical Overview.** *Anthropic Research Papers*.
