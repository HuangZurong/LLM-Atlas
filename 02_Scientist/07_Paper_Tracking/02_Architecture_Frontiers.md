# Architecture Frontiers

*Prerequisite: [../01_Architecture/](../01_Architecture/). Continuously updated. Last update: 2025-06*

---

## 1. SSM / Hybrid Architectures

### [Mamba-2] Transformers are SSMs (Dao & Gu, 2024)
- **Institution**: Carnegie Mellon / Princeton
- **Link**: arXiv:2405.21060
- **One-line Summary**: Proves that linear attention variants of Transformers are equivalent to SSMs; proposes the SSD (Structured State Space Duality) layer, 2-8× faster than Mamba-1.
- **Core Innovation**: Unifies the recurrent form of SSMs and the matrix form of attention into a single framework, allowing free switching between computation modes.
- **Key Results**: Matches Transformer++ quality on language modeling with 2-8× training throughput improvement.
- **Limitations**: Still weaker than standard Attention on in-context learning and precise retrieval tasks.

### [Jamba] Jamba: A Hybrid Transformer-Mamba Language Model (AI21, 2024)
- **Institution**: AI21 Labs
- **Link**: arXiv:2403.19887
- **One-line Summary**: First large-scale Transformer-Mamba hybrid model (52B total params, 12B active), interleaving Transformer and Mamba layers + MoE.
- **Core Innovation**: Demonstrates hybrid architecture viability at industrial scale — 256K context window, inference on a single 80GB GPU.
- **Relation to Existing Work**: Validates the engineering feasibility of the Hybrid route.

---

## 2. Attention Mechanism Innovations

### [MLA] DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model (2024)
- **Institution**: DeepSeek
- **Link**: arXiv:2405.04434
- **One-line Summary**: Proposes Multi-head Latent Attention (MLA), reducing KV Cache by 93.3% through low-rank joint compression.
- **Core Innovation**: Compresses KV into a low-dimensional latent vector, with Decoupled RoPE to maintain positional encoding compatibility.
- **Key Results**: 236B total / 21B active params; matches Llama 3 70B performance at significantly lower inference cost.
- **Implications**: Updated in 01_Architecture/02_Attention.md.

### [NSA] Native Sparse Attention (DeepSeek, 2025)
- **Institution**: DeepSeek
- **Link**: arXiv:2502.11089
- **One-line Summary**: Hardware-aligned sparse attention decomposed into compression tokens + selected tokens + sliding window branches.
- **Core Innovation**: Sparsity patterns are learned during training (not hand-designed) and aligned to hardware block sizes for real speedup.
- **Key Results**: 6-9× forward pass speedup at 64K context while maintaining full-attention quality.
- **Limitations**: Only validated internally at DeepSeek; community reproduction pending.

---

## 3. Long Context

### [Ring Attention] Ring Attention with Blockwise Transformers (Liu et al., 2023)
- **Institution**: UC Berkeley
- **Link**: arXiv:2310.01889
- **One-line Summary**: Distributes attention computation across devices, each holding one KV block, achieving near-infinite context via ring communication.
- **Core Innovation**: Combines blockwise attention with ring communication; context length scales linearly with device count.
- **Relation to Existing Work**: Complementary to FlashAttention (FA optimizes single-device; Ring Attention optimizes cross-device).

### [YaRN] YaRN: Efficient Context Window Extension (Peng et al., 2023)
- **Institution**: EleutherAI / Nous Research
- **Link**: arXiv:2309.00071
- **One-line Summary**: Improved RoPE extrapolation via NTK-aware interpolation + attention scaling for efficient context extension.
- **Core Innovation**: Applies different interpolation strategies to different RoPE frequency dimensions (extrapolate low-frequency, preserve high-frequency).
- **Key Results**: Extends Llama 2 from 4K to 128K context with only ~400 fine-tuning steps.

---

## 4. Frontier Model Architectures (2025)

### [Llama 4] Llama 4: Scout, Maverick, and Behemoth (Meta, 2025)
- **Institution**: Meta
- **Link**: arXiv:2601.11659
- **One-line Summary**: First MoE-based Llama family; Scout (17B active / 109B total) supports 10M-token context, Maverick (17B active / 400B total) for general tasks, Behemoth (288B active / 2T total) as teacher.
- **Core Innovation**: MoE routing + iRoPE (interleaved RoPE with no-position layers) for extreme context extension; Early Fusion for native multimodal (text + image) processing from pre-training.
- **Key Results**: Scout achieves 10M context window; Maverick matches GPT-4o on reasoning benchmarks.
- **Implications**: Validates MoE + native multimodal as the dominant open-weight architecture direction.

### [Qwen3] Qwen3: Thinking Deeper, Faster, and More Efficiently (Alibaba, 2025)
- **Institution**: Alibaba Qwen
- **Link**: qwen3 technical blog
- **One-line Summary**: MoE family from 0.6B to 235B (22B active), featuring "thinking mode" toggle for hybrid reasoning.
- **Core Innovation**: 235B-A22B uses 128 experts with top-8 routing + GQA; supports seamless switching between "thinking" (long CoT) and "non-thinking" (fast) modes.
- **Key Results**: Qwen3-235B-A22B matches DeepSeek-R1 on AIME 2024; Qwen3-30B-A3B outperforms QwQ-32B at 1/10 active params.

### [GPT-4.1] GPT-4.1 Family (OpenAI, 2025)
- **Institution**: OpenAI
- **Link**: openai.com/index/gpt-4-1
- **One-line Summary**: API-only model family (4.1 / Mini / Nano) with 1M-token context, major coding and instruction-following improvements.
- **Core Innovation**: 1M-token context window; optimized for agentic workflows and long-document understanding.
- **Key Results**: SWE-bench 54.6% (+21.4% over GPT-4o); significantly improved instruction following and structured output reliability.
