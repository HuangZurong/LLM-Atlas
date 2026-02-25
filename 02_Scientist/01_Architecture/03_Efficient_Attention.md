# Efficient Attention & Alternative Architectures

*Prerequisite: [02_Attention.md](02_Attention.md).*

Standard Transformer attention scales as $O(N^2)$ in both compute and memory. This document covers the major approaches to breaking that bottleneck — from hardware-aware exact attention to sub-quadratic alternatives.

---

## 1. FlashAttention

### 1.1 Core Idea

FlashAttention (Dao et al., 2022) does not change the math — it computes exact attention. The innovation is a **hardware-aware tiling algorithm** that minimizes HBM (High Bandwidth Memory) reads/writes by keeping intermediate results in fast GPU SRAM.

### 1.2 Key Techniques

| Technique | Description |
|:--|:--|
| **Tiling** | Splits Q, K, V into blocks that fit in SRAM; computes partial softmax per block |
| **Online Softmax** | Maintains running max and sum to compute exact softmax across tiles without materializing the full $N \times N$ attention matrix |
| **Recomputation** | During backward pass, recomputes attention from Q, K, V instead of storing the $O(N^2)$ matrix — trades FLOPs for memory |

### 1.3 Complexity

- **FLOPs**: $O(N^2 d)$ — unchanged from standard attention.
- **Memory**: $O(N)$ — avoids materializing the $N \times N$ attention matrix.
- **Wall-clock speedup**: 2–4× over PyTorch standard attention due to reduced HBM I/O.

### 1.4 FlashAttention-2 & 3

- **FA-2 (Dao, 2023)**: Better work partitioning across warps; ~2× faster than FA-1.
- **FA-3 (Shah et al., 2024)**: Exploits Hopper GPU features (TMA, WGMMA, FP8); further speedups on H100.

---

## 2. Linear Attention

### 2.1 Motivation

Replace the softmax kernel $\text{softmax}(QK^T)$ with a decomposable kernel $\phi(Q)\phi(K)^T$, enabling the computation to be rewritten as:

$$\text{Attn}(Q, K, V) = \phi(Q) \left( \phi(K)^T V \right)$$

The inner term $\phi(K)^T V \in \mathbb{R}^{d \times d}$ can be computed once and reused, yielding $O(Nd^2)$ complexity — linear in sequence length.

### 2.2 Limitations

- **Approximation quality**: The kernel approximation degrades for tasks requiring sharp, precise attention patterns (e.g., retrieval, copying).
- **In practice**: Pure linear attention models underperform Transformers on most benchmarks. The approach is more successful as a component in hybrid architectures.

---

## 3. State Space Models (SSMs)

### 3.1 Structured State Spaces (S4)

Maps input sequences through a continuous-time linear system discretized for efficient computation:

$$h'(t) = Ah(t) + Bx(t), \quad y(t) = Ch(t) + Dx(t)$$

Key insight: the recurrence can be computed as a **convolution** during training (parallelizable) and as a **recurrence** during inference (constant memory per step).

### 3.2 Mamba (Selective SSM)

Gu & Dao (2023) introduced **input-dependent selection** — the matrices $B$, $C$, and the discretization step $\Delta$ are functions of the input, allowing the model to selectively propagate or forget information.

| Property | Transformer | Mamba |
|:--|:--|:--|
| Training complexity | $O(N^2 d)$ | $O(N d^2)$ (linear in $N$) |
| Inference (per step) | $O(Nd)$ (KV cache lookup) | $O(d^2)$ (fixed-size state) |
| Long-range retrieval | Exact (full attention) | Lossy (compressed state) |

### 3.3 The Compression Trade-off

SSMs compress the entire history into a fixed-size hidden state. This is efficient but fundamentally lossy — for "needle in a haystack" retrieval tasks, Transformers with exact attention remain superior.

---

## 4. RWKV

Combines Transformer-style parallel training with RNN-style constant-memory inference:

- **Time-Mixing block**: Replaces attention with a linear recurrence using learned decay factors.
- **Channel-Mixing block**: Replaces FFN with a similar gated structure.
- **Complexity**: $O(Nd)$ for both training and inference.
- **Trade-off**: Similar to Mamba — efficient but with reduced precision on retrieval-heavy tasks.

---

## 5. Hybrid Architectures

The emerging consensus (2024–2025) is that **pure alternatives rarely beat Transformers**, but **hybrids** can capture the best of both worlds:

| Architecture | Design | Example |
|:--|:--|:--|
| **Mamba-Transformer** | Interleave Mamba layers with full attention layers | Jamba (AI21, 2024) |
| **SSM + Sliding Window** | SSM for long-range, local attention for precision | StripedHyena |
| **Gated Linear Attention** | Linear attention with data-dependent gating | GLA (Yang et al., 2024) |

---

## 6. Decision Framework

| Scenario | Recommended Architecture |
|:--|:--|
| General-purpose LLM (quality-first) | Transformer + FlashAttention |
| Very long context (>128K) with retrieval needs | Hybrid (SSM + sparse attention) |
| Edge / low-memory deployment | Mamba or RWKV |
| Throughput-critical batch inference | FlashAttention-3 on Hopper GPUs |

---

## 7. Key References

1. **Dao et al. (2022)**: *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*.
2. **Dao (2023)**: *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*.
3. **Gu et al. (2022)**: *Efficiently Modeling Long Sequences with Structured State Spaces* (S4).
4. **Gu & Dao (2023)**: *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*.
5. **Peng et al. (2023)**: *RWKV: Reinventing RNNs for the Transformer Era*.
6. **Lieber et al. (2024)**: *Jamba: A Hybrid Transformer-Mamba Language Model*.
7. **Shah et al. (2024)**: *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*.
