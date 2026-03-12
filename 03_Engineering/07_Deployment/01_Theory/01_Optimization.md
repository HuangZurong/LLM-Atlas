# Model Deployment & Inference Optimization

*Prerequisite: [../../01_LLMs/01_Theory/03_API_Mechanics.md](../../01_LLMs/01_Theory/03_API_Mechanics.md).*
*See Also: [../../../04_Solutions/02_Cost_ROI_Analysis_Model.md](../../../04_Solutions/02_Cost_ROI_Analysis_Model.md) (cost analysis & break-even), [../../../02_Scientist/03_Pre_Training/09_Research_Trends.md](../../../02_Scientist/03_Pre_Training/09_Research_Trends.md) (efficiency trends).*

---

Once a model is trained or fine-tuned, the focus shifts to **Inference Engineering**. Serving 70B+ models effectively requires moving beyond the standard HuggingFace `generate()` loop.

## 1. Inference Bottlenecks

- **Memory Bandwidth**: The primary bottleneck for auto-regressive generation.
- **Compute Bound**: Less common for LLMs, but occurs during the **Prefill** phase (initial prompt processing).
- **VRAM Utilization**: KV Cache growth limits batch size and context length.

## 2. vLLM & PagedAttention [Industry Standard]

The breakthrough for serving LLMs in production.

- **Problem**: Standard KV caching fragments memory (up to 80% waste).
- **Solution**: **PagedAttention** partitions the KV cache into non-contiguous physical blocks, similar to virtual memory in operating systems.
- **Impact**: Increases throughput by 2x-5x compared to standard serving.

## 3. Advanced Optimization Techniques

- **Speculative Decoding**: Using a small "draft model" (e.g., TinyLlama) to predict multiple tokens at once, while the "target model" (e.g., Llama-3-70B) verifies them in parallel.
- **Continuous Batching**: Decouples the request batching from the iteration loop, allowing new requests to join without waiting for current ones to finish.
- **FlashAttention-3**: Optimization for Newer NVIDIA H100 GPUs to speed up the core attention calculation.

---

[vLLM: High-throughput serving with PagedAttention](https://arxiv.org/abs/2309.06180)
