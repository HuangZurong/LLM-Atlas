# LLM Deployment Architecture

*Prerequisite: [01_Optimization.md](01_Optimization.md).*

---

## 1. Deployment Patterns

How you structure your model serving determines your quality-cost-latency balance.

### 1.1 Single-Model Serving
The simplest pattern: an API gateway forwards all requests to a single inference engine running one model.
- **Pros**: Simple to maintain, consistent behavior.
- **Cons**: Suboptimal cost (using GPT-4 for "hello") or quality (using Qwen-7B for complex reasoning).

### 1.2 Model Routing (Router Pattern)
A lightweight classifier (or rules) routes the query to the most suitable model.
- **Easy task** → GPT-4o-mini / Llama-8B (fast, cheap)
- **Hard task** → GPT-4o / DeepSeek-V3 (slow, expensive)
- **Coding task** → DeepSeek-Coder-V2

### 1.3 Cascade (Small-to-Large)
First call a small model. If its confidence score is low or it fails a validation check, escalate to a larger model.
- **Pros**: Maximizes efficiency on simple queries.
- **Cons**: Adds latency for complex queries due to the double-call.

### 1.4 Ensemble / MoE (Mixture of Experts)
Query multiple models in parallel and use a judge model (or voting) to synthesize the final answer.
- **Pros**: Highest possible quality, reduces variance and hallucination.
- **Cons**: highest cost and latency.

## 2. Serving Infrastructure

### 2.1 Inference Engines
| Engine | Key Innovation | Best Use Case |
|---|---|---|
| **vLLM** | PagedAttention | High-throughput production serving |
| **TGI (HuggingFace)** | Flash Attention, continuous batching | Deploying HF models quickly |
| **SGLang** | RadixAttention (prefix caching) | Complex prompt templates, many shared prefixes |
| **Ollama / llama.cpp** | GGUF quantization | Local development, CPU inference |

### 2.2 Orchestration
- **Docker + Kubernetes**: Standard for scaling model replicas.
- **NVIDIA GPU Operator**: Simplifies GPU resource management in K8s.
- **KServe / Ray Serve**: Dedicated frameworks for scaling AI workloads on K8s.

### 2.3 Autoscaling Strategies
- **Queue Depth**: Scale when the number of pending requests exceeds a threshold.
- **GPU Utilization**: Scale when average VRAM or GPU compute usage is high.
- **Latency (TTFT)**: Scale if time-to-first-token exceeds user experience targets.

## 3. API Design

### 3.1 Response Patterns
- **Streaming (SSE)**: Send tokens as they are generated. Essential for user-facing chat.
- **Non-Streaming (Unary)**: Send the full response once generation is complete. Best for background tasks/APIs.
- **Batch API**: Submit a file of prompts, receive results 6-24 hours later at 50% discount.

### 3.2 Security & Multi-tenancy
- **Authentication**: JWT/API keys for user identity.
- **Rate Limiting**: Tiered access (e.g., 5 RPM for free tier, 500 RPM for pro).
- **Usage Tracking**: Monitor token consumption by user/org for billing.

## 4. Caching Strategies

Caching is the most effective way to reduce cost and latency.

| Type | How it Works | Key Benefit |
|---|---|---|
| **Prompt Caching** | Cache KV cache blocks for shared prefixes (system prompts) | 50–90% faster TTFT, lower cost |
| **Exact Match** | MD5(prompt + params) → cached result | Near-zero cost/latency for repeated queries |
| **Semantic Caching** | Vector similarity search: if current query is semantically similar to a cached one, return cached result | High hit rate for common user intents |

**Frameworks**: GPTCache, LangChain Cache.

## 5. Cost Optimization

### 5.1 Quantization
Reducing model precision to save VRAM and increase throughput.
- **GPTQ / AWQ**: 4-bit quantization for production GPUs.
- **GGUF**: Best for mixed CPU/GPU or edge inference.
- **FP8**: Native support in newer H100/L40S GPUs.

### 5.2 Model Distillation
Training a smaller "student" model (e.g., 8B) to mimic the behavior of a "teacher" model (e.g., 405B).
- **Result**: You get 80–90% of the teacher's capability at 1/20th the inference cost.

### 5.3 Batching Optimization
- **Continuous Batching**: Allows new requests to join the running batch without waiting for others to finish.
- **Dynamic Batching**: Groups requests into a batch only when they arrive close together.

## 6. Reliability & Resilience

- **Health Checks**: Monitor `/health` endpoints of inference engines.
- **Fallback Chains**:
  1. Primary Model (DeepSeek-V3)
  2. Fallback Model (GPT-4o)
  3. Static Response ("System is busy, please try later")
- **Graceful Degradation**: If system load is too high, automatically disable expensive features (like Vision or Search) or switch to a faster, smaller model.
- **Request Shadowing**: Send a copy of production traffic to a new model version for testing without affecting users.
