# Virtualization & Quantization: Serving at Scale

*Prerequisite: [01_Optimization.md](01_Optimization.md).*

---

To serve large models efficiently, you need to minimize memory footprint and maximize hardware utilization. This document covers the two most effective techniques: **model quantization** and **inference engine optimization**.

## 1. Quantization: Precision vs. Performance

Quantization reduces model precision to save memory and increase throughput. The trade-off is subtle quality loss vs. dramatic speedup.

### 1.1 Quantization Techniques

| Method | Bits | Memory Savings | Quality Drop | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **FP16** | 16-bit | 2× vs. FP32 | ~0.1% | General production serving |
| **BF16** | 16-bit | 2× vs. FP32 | ~0.1% | Modern NVIDIA GPUs (Ampere+) |
| **INT8** | 8-bit | 4× vs. FP32 | 0.5-2% | High-throughput inference |
| **GPTQ** | 4-bit | 8× vs. FP32 | 1-3% | Consumer GPUs, edge devices |
| **AWQ** | 4-bit | 8× vs. FP32 | 0.5-2% | Activation-aware optimization |
| **GGUF** | 2-8 bit | 4-16× | 1-5% | CPU inference, Apple Silicon |
| **FP8** | 8-bit | 2× vs. FP16 | ~0.2% | H100/L40S native support |

### 1.2 Quantization Decision Tree

```
Model Size? ──┬─ <7B → FP16/BF16 (no quantization needed)
              ├─ 7-70B → INT8/AWQ (good balance)
              ├─ 70B+ → GPTQ/AWQ (required for single GPU)
              └─ CPU/Edge → GGUF (2-5 bit)
```

**Rule of thumb**: Start with 8-bit (INT8). If you need more memory savings, go to 4-bit (GPTQ/AWQ). Only use 2-bit for extreme constraints.

## 2. Inference Engine Virtualization

Modern inference engines use virtual memory techniques to optimize KV cache management.

### 2.1 PagedAttention (vLLM)

The breakthrough that made high-throughput serving possible.

**Problem**: Standard KV caching creates **memory fragmentation**:
- Each request reserves memory for its full context window
- Even if actual context is shorter, memory is wasted
- Fragmentation reaches 60-80% in production

**Solution**: PagedAttention treats KV cache like OS virtual memory:
- KV cache is divided into **blocks** (e.g., 16 tokens per block)
- Blocks are allocated **non-contiguously** in physical memory
- Requests share blocks when possible (prefix caching)

**Impact**: 2-5× throughput improvement vs. naive caching.

### 2.2 RadixAttention (SGLang)

Optimization for **prompt templates** with shared prefixes.

**Problem**: Many requests start with the same system prompt (e.g., "You are a helpful assistant...").

**Solution**: RadixAttention builds a **prefix tree** of common prompt patterns:
- Shared prefixes are computed **once** and cached
- New requests starting with cached prefix skip prefill computation
- 50-90% reduction in time-to-first-token (TTFT)

**Best for**: Chat applications with standardized system prompts.

### 2.3 Continuous Batching

Traditional batching groups requests that arrive at the same time. Continuous batching groups requests **dynamically** during generation.

```
Traditional Batching:
Request A: ████████████████████████ (full sequence)
Request B: ████████████████████████ (must wait for A)

Continuous Batching:
Request A: ████████████
Request B:        ██████████████ (joins mid-way)
Request C:               █████████ (joins later)
```

**Key metrics**:
- **Batching Efficiency** = tokens_generated / tokens_computed
- Target: >0.7 (70% of compute used for useful tokens)

## 3. Inference Engine Comparison (2025)

| Engine | Core Innovation | Throughput | Memory Efficiency | Ease of Use |
| :--- | :--- | :--- | :--- | :--- |
| **vLLM** | PagedAttention | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **TGI** (HuggingFace) | Flash Attention, Continuous Batching | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **SGLang** | RadixAttention | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **TensorRT-LLM** | NVIDIA-optimized kernels | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **llama.cpp** | GGUF, CPU-first | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Ollama** | Desktop-friendly wrapper | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### 3.1 vLLM Production Configuration

```python
# Typical vLLM deployment config
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,           # Split across 4 GPUs
    gpu_memory_utilization=0.9,       # Use 90% of VRAM
    max_num_seqs=256,                 # Maximum concurrent requests
    max_num_batched_tokens=8192,      # Batch size limit
    enable_prefix_caching=True,       # Radix-like optimization
    block_size=16,                    # PagedAttention block size
    swap_space=4,                     # 4GB swap for CPU offloading
)
```

### 3.2 TGI (HuggingFace Text Generation Inference)

```bash
# Docker deployment
docker run --gpus all \
  -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference:2.0 \
  --model-id meta-llama/Llama-3.1-70B-Instruct \
  --quantize bitsandbytes-nf4 \
  --max-input-length 32768 \
  --max-total-tokens 32768 \
  --max-batch-prefill-tokens 8192
```

## 4. Hardware Considerations

### 4.1 GPU Selection Matrix

| GPU | VRAM | FP8 | BF16 | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **H100** | 80GB | ✅ | ✅ | Training, large-batch inference |
| **L40S** | 48GB | ✅ | ✅ | Cost-effective inference |
| **A100** | 40/80GB | ❌ | ✅ | General-purpose |
| **RTX 4090** | 24GB | ❌ | ✅ | Development, small models |
| **RTX 3090** | 24GB | ❌ | ✅ | Budget development |

### 4.2 Multi-GPU Strategies

| Strategy | Implementation | Use Case |
| :--- | :--- | :--- |
| **Tensor Parallelism** | Model layers split across GPUs | Single large model (<1s latency) |
| **Pipeline Parallelism** | Model stages split across GPUs | Very large models (>100B) |
| **Model Sharding** | Different models on different GPUs | Multi-model serving |
| **Hybrid** | TP + PP + DP | Maximum scale |

## 5. Performance Benchmarks

### 5.1 Llama-3.1-70B on H100 (vLLM)

| Batch Size | Tokens/sec | VRAM Usage | Latency (p95) |
| :--- | :--- | :--- | :--- |
| 1 | 120 | 40GB | 850ms |
| 8 | 850 | 65GB | 1.2s |
| 32 | 2,800 | 78GB | 2.8s |
| 128 | 8,500 | 78GB | 8.5s |

### 5.2 Quantization Impact (GPT-4-class model)

| Precision | Throughput | Quality (MMLU) | VRAM (70B) |
| :--- | :--- | :--- | :--- |
| FP16 | 1.0× (baseline) | 85.2 | 140GB |
| INT8 | 1.8× | 84.7 | 70GB |
| GPTQ (4-bit) | 2.5× | 82.1 | 35GB |
| AWQ (4-bit) | 2.3× | 83.5 | 35GB |

## 6. Practical Guidelines

1. **Start with vLLM** for production serving (best balance of features)
2. **Use 8-bit quantization** (INT8) as default for models >7B
3. **Enable prefix caching** if you have standardized prompts
4. **Monitor GPU memory utilization** — target 80-90% for efficiency
5. **Test quantization on your domain** — quality loss varies by task
6. **Consider hybrid deployment** — CPU for preprocessing, GPU for inference