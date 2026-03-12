# Cost Optimization Strategies for LLM Deployment

*Prerequisite: [../01_Theory/04_Cloud_Platforms_Comparison.md](../01_Theory/04_Cloud_Platforms_Comparison.md).*

---

LLM inference can consume 70-90% of AI infrastructure costs in production. This document outlines proven strategies to reduce costs while maintaining quality and latency SLAs.

## 1. Cost Breakdown Analysis

### 1.1 Where the Money Goes

```
Typical LLM Application Cost Breakdown:

┌─────────────────────────────────────────────────┐
│ LLM API Calls                   50-70%          │
│   - OpenAI/Anthropic API costs                  │
│   - Self-hosted GPU costs                       │
├─────────────────────────────────────────────────┤
│ Infrastructure                  20-30%          │
│   - GPU instances                              │
│   - Kubernetes clusters                         │
│   - Network egress                              │
├─────────────────────────────────────────────────┤
│ Data Processing & Storage       10-15%          │
│   - Vector databases                            │
│   - Object storage (S3)                         │
│   - Logging & monitoring                        │
├─────────────────────────────────────────────────┤
│ Development & Operations         5-10%          │
│   - CI/CD pipelines                            │
│   - Security scanning                           │
│   - Team overhead                               │
└─────────────────────────────────────────────────┘
```

### 1.2 Key Cost Drivers

| Driver | Impact | Typical Range |
| :--- | :--- | :--- |
| **Model Size** | Exponential cost increase with parameters | 7B: $0.01/req → 70B: $0.10/req |
| **Context Length** | KV cache memory grows linearly | 4K: 1× → 128K: 32× memory |
| **Request Volume** | Linear scaling with traffic | 1K req/day: $10 → 1M req/day: $10K |
| **Token Count** | Input + output tokens billed | Avg 500 tokens/request |
| **GPU Type** | H100 vs A100 vs consumer GPU | H100: $4.76/hr vs 4090: $0.79/hr |

## 2. Model Selection Optimization

### 2.1 Tiered Model Strategy

**Pattern**: Use different models for different query complexities.

| Query Type | Recommended Model | Cost vs GPT-4o |
| :--- | :--- | :--- |
| **Simple Q&A** | GPT-4o-mini / Claude Haiku | 5-10% |
| **General Reasoning** | GPT-4o / Claude Sonnet | 100% (baseline) |
| **Complex Analysis** | GPT-4o-2024-08 / Claude Opus | 200-300% |
| **Code Generation** | DeepSeek-Coder / Codestral | 20-40% |

**Implementation**: Use `03_Model_Router.py` to route queries automatically.

### 2.2 Cascade Fallback Pattern

```python
async def generate_with_cascade(query: str) -> str:
    # Try cheap model first
    try:
        response = await cheap_model.generate(query)
        confidence = await evaluate_confidence(response, query)

        if confidence > 0.8:
            return response  # Cheap model succeeded
    except Exception:
        pass

    # Fall back to expensive model
    return await expensive_model.generate(query)
```

**Impact**: 60-80% of queries served by cheap models, saving 40-60% on costs.

### 2.3 Model Distillation

Train smaller "student" models to mimic larger "teacher" models:

| Student Model | Teacher Model | Quality Retention | Cost Reduction |
| :--- | :--- | :--- | :--- |
| Llama-3.1-8B | Llama-3.1-70B | 85-90% | 80-90% |
| Qwen2.5-7B | Qwen2.5-72B | 80-85% | 85-90% |
| DistilBERT | BERT-large | 95% | 60-70% |

**Use case**: Domain-specific applications where you can fine-tune a small model.

## 3. Inference Optimization

### 3.1 Quantization Techniques

| Technique | Bits | Memory Savings | Quality Drop | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **FP16** | 16 | 2× | <0.5% | General production |
| **INT8** | 8 | 4× | 1-2% | High-throughput |
| **GPTQ** | 4 | 8× | 2-5% | Single GPU deployment |
| **AWQ** | 4 | 8× | 1-3% | Activation-aware |
| **GGUF (Q4_K_M)** | 4-5 | 5-7× | 3-7% | CPU/Apple Silicon |

**Decision rule**: Start with 8-bit, move to 4-bit only if memory constrained.

### 3.2 Continuous Batching

**Problem**: Traditional batching wastes GPU cycles when requests complete at different times.

**Solution**: Continuous batching allows new requests to join mid-generation.

```
Traditional: ██████████████████ (wasted capacity)
Continuous:  ████████ ██████████ (full utilization)
```

**Impact**: 2-5× throughput improvement, 50-80% cost reduction per token.

### 3.3 KV Cache Optimization

**PagedAttention (vLLM)**: Eliminates 60-80% memory fragmentation in KV cache.

**Impact**: Serve 2× more concurrent requests on same hardware.

## 4. Caching Strategies

### 4.1 Multi-Layer Cache Architecture

```
┌─────────────────────────────────────────────────┐
│ Layer 1: Exact Match Cache (Redis)              │
│   - MD5(prompt + params) → response             │
│   - Hit rate: 10-30%                            │
│   - Latency: <5ms                               │
├─────────────────────────────────────────────────┤
│ Layer 2: Semantic Cache (Vector DB)             │
│   - Embedding similarity >0.85 → cached response│
│   - Hit rate: 20-40%                            │
│   - Latency: 20-50ms                            │
├─────────────────────────────────────────────────┤
│ Layer 3: Template Cache                         │
│   - Parameterized queries (e.g., weather in {city})
│   - Hit rate: 5-15%                             │
│   - Latency: 10-30ms                            │
└─────────────────────────────────────────────────┘
```

**Overall hit rate**: 35-85% depending on query distribution.

### 4.2 Cache Configuration Best Practices

| Setting | Recommendation | Rationale |
| :--- | :--- | :--- |
| **TTL** | 1 hour for dynamic, 24 hours for static | Balance freshness vs hit rate |
| **Max Size** | 10-20% of expected daily tokens | Prevent cache bloat |
| **Eviction** | LRU (Least Recently Used) | Natural prioritization |
| **Warm-up** | Pre-cache frequent queries | Reduce cold start penalty |
| **Invalidation** | Version-based for model updates | Ensure cache consistency |

## 5. Infrastructure Optimization

### 5.1 Cloud Platform Selection

| Provider | GPU Cost (A100-80GB/hr) | Best For |
| :--- | :--- | :--- |
| **CoreWeave** | $2.76 | Cost-sensitive production |
| **Lambda Labs** | $1.10 (spot) | Development, batch jobs |
| **AWS** | $32.77 | Enterprise compliance |
| **GCP** | $23.44 | Google ecosystem |
| **Azure** | $32.77 | Microsoft shops |

**Strategy**: Use cheap providers for steady-state, hyperscalers for compliance needs.

### 5.2 Autoscaling Configuration

```yaml
# Kubernetes HPA configuration for GPU workloads
metrics:
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 70  # Scale at 70% GPU util
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80  # Scale at 80% memory
```

**Target utilization**: 70-80% for GPU, 80-90% for memory.

### 5.3 Spot/Preemptible Instances

**Savings**: 70-90% discount for interruptible workloads.

**Use cases**:
- Batch inference jobs
- Model training/fine-tuning
- Development and testing
- Non-critical background tasks

**Pattern**: Warm pool of on-demand instances + spot for burst capacity.

## 6. Token Economics

### 6.1 Token Reduction Techniques

| Technique | Reduction | Impact |
| :--- | :--- | :--- |
| **Prompt Compression** | 20-50% | Minimal quality loss |
| **Output Token Limits** | 30-70% | Speed + cost improvement |
| **Context Window Management** | 40-60% | Major memory savings |
| **Stop Sequences** | 5-20% | Prevent verbose outputs |

### 6.2 Efficient Prompt Design

**Bad**: "Please provide a comprehensive analysis of the following topic: {topic}"

**Good**: "Analyze {topic}. Key points only."

**Impact**: 50% token reduction, same quality for many tasks.

### 6.3 Streaming vs. Batch

| Mode | Cost Efficiency | Use Case |
| :--- | :--- | :--- |
| **Streaming** | Lower (user can stop early) | Interactive chat |
| **Batch API** | 50% discount | Offline processing |
| **Sync Completion** | Baseline | API integrations |

## 7. Monitoring & Cost Attribution

### 7.1 Key Metrics to Track

| Metric | Target | Alert Threshold |
| :--- | :--- | :--- |
| **Cost per Request** | <$0.05 | >$0.10 |
| **Tokens per Dollar** | >2000 | <1000 |
| **Cache Hit Rate** | >40% | <20% |
| **GPU Utilization** | 70-90% | <50% or >95% |
| **Model Routing Accuracy** | >90% | <80% |

### 7.2 Cost Attribution

Implement per-feature, per-team, per-customer cost tracking:

```sql
-- Example cost attribution schema
requests
├── request_id
├── customer_id
├── feature_name
├── model_used
├── input_tokens
├── output_tokens
├── cost_usd
├── cache_hit (bool)
└── timestamp
```

**Use cases**: Showback/chargeback, feature ROI analysis, abuse detection.

## 8. Implementation Roadmap

### 8.1 Quick Wins (Week 1)

1. **Add caching layer**: 20-40% immediate savings
2. **Implement model routing**: 30-50% savings on simple queries
3. **Set token limits**: 10-30% reduction in verbose outputs
4. **Enable quantization**: 50% memory reduction for self-hosted

### 8.2 Medium-term (Month 1)

1. **Deploy to cost-optimized cloud**: 30-70% infrastructure savings
2. **Implement continuous batching**: 2-5× throughput improvement
3. **Add cascade fallback**: 40-60% savings on mixed complexity workloads
4. **Set up cost monitoring**: Identify optimization opportunities

### 8.3 Long-term (Quarter 1)

1. **Model distillation**: 80-90% cost reduction for domain-specific tasks
2. **Multi-cloud strategy**: Leverage cheapest provider for each workload
3. **Predictive scaling**: Match capacity to traffic patterns
4. **Custom kernels**: Hardware-specific optimizations

## 9. Trade-off Considerations

| Optimization | Cost Savings | Quality Impact | Complexity |
| :--- | :--- | :--- | :--- |
| **4-bit Quantization** | 70-80% | 2-5% ↓ | Medium |
| **Model Routing** | 40-60% | 0-2% ↓ | Low |
| **Caching** | 35-85% | 0% (if fresh) | Low |
| **Continuous Batching** | 50-80% | 0% | High |
| **Spot Instances** | 70-90% | Reliability risk | Medium |

**Rule**: Start with high-impact, low-complexity optimizations. Move to complex optimizations only when justified by scale.

## 10. Success Metrics

- **Cost per MAU** < $0.50 (monthly active user)
- **Tokens per Dollar** > 2000
- **Cache Hit Rate** > 40%
- **GPU Utilization** > 70%
- **Model Routing Accuracy** > 90%

**Benchmark**: Well-optimized LLM applications achieve 80-90% cost reduction compared to naive GPT-4 API usage.