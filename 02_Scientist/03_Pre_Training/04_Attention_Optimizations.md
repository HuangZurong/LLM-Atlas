# Attention Optimizations: Theory, Kernels, and Scalability

*Prerequisite: [../01_Architecture/02_Attention.md](../01_Architecture/02_Attention.md), [../01_Architecture/03_Efficient_Attention.md](../01_Architecture/03_Efficient_Attention.md).*

---

## 1. The Attention Bottleneck: Memory and Compute

### 1.1 Complexity Breakdown
Standard Self-Attention (Vaswani et al., 2017) for sequence length $N$ and head dimension $d$:

| Phase | Operation | Complexity | Memory (HBM) |
|:--|:--|:--|:--|
| **Projections** | $Q, K, V = XW_Q, XW_K, XW_V$ | $O(Nd^2)$ | $O(Nd)$ |
| **Scores** | $S = QK^\top$ | $O(N^2d)$ | $O(N^2)$ (Bottleneck!) |
| **Softmax** | $P = \text{softmax}(S / \sqrt{d})$ | $O(N^2)$ | $O(N^2)$ |
| **Context** | $O = PV$ | $O(N^2d)$ | $O(Nd)$ |

**The $O(N^2)$ Memory Wall**: For $N=128K$, the score matrix $S$ requires **64GB** of HBM memory (FP16) for a single head. Distributed training is impossible without optimization.

---

## 2. FlashAttention: IO-Aware Tiling

### 2.1 The IO Bottleneck Theory
Modern GPUs (A100/H100) have massive compute (TFLOPS) but limited HBM bandwidth.
- **HBM (Slow)**: Large capacity (80GB), slow access.
- **SRAM (Fast)**: Small capacity (20MB), extremely fast access.

**Goal**: Avoid writing the $N^2$ matrix back to HBM.

### 2.2 Tiling and Online Softmax
FlashAttention computes attention by blocks (tiles) that fit into SRAM.

#### 2.2.1 Online Softmax Formula
To compute softmax over tiles without the full denominator:
Let $x^{(1)}, x^{(2)}$ be two tiles.
$$m(x) = \max(x), \quad L(x) = \sum e^{x_i - m(x)}$$

Merged statistics for $[x^{(1)}, x^{(2)}]$:
$$m_{new} = \max(m^{(1)}, m^{(2)})$$
$$L_{new} = L^{(1)} e^{m^{(1)} - m_{new}} + L^{(2)} e^{m^{(2)} - m_{new}}$$

#### 2.2.2 FlashAttention-2 Algorithm (Tri Dao, 2023)
1. **Partition** $Q$ into blocks $Q_i$ (row-wise), $K, V$ into blocks $K_j, V_j$ (column-wise).
2. **Loop $j$ (outer)**: Load $K_j, V_j$ to SRAM.
3. **Loop $i$ (inner)**: Load $Q_i$ to SRAM.
   - Compute $S_{ij} = Q_i K_j^\top$.
   - Update running softmax statistics $m_i, L_i$.
   - Update partial output $O_i$.
4. **Result**: 2-4x speedup, memory reduction from $O(N^2)$ to $O(N)$.

---

## 3. KV Cache Architectural Optimizations

### 3.1 Multi-Query Attention (MQA)
One set of $K, V$ heads shared by all $Q$ heads.

**KV Memory reduction factor**: $H$ (number of heads).
**Equation**:
$$\text{Head}_i = \text{Attn}(X W_{Q,i}, X W_{K, \text{shared}}, X W_{V, \text{shared}})$$

### 3.2 Grouped-Query Attention (GQA)
A compromise used in **Llama 3** and **Mistral**. Heads are divided into $G$ groups. Each group shares one $K$ and $V$ head.

**KV Memory reduction factor**: $H / G$.
**Standard Configuration**: $H=32, G=8 \Rightarrow 4\times$ reduction.

### 3.3 Sliding Window Attention (SWA)
Used in **Mistral 7B**. Tokens only attend to the last $W$ tokens.

**Effective Complexity**: $O(N \times W)$.
**Rolling Buffer Cache**: Instead of growing the cache indefinitely, overwrite old tokens using modulo indexing: `pos % W`.

---

## 4. Systems Optimization: PagedAttention (vLLM)

### 4.1 Memory Fragmentation
Standard KV caching allocates a fixed contiguous block for each request.
- **Problem**: Requests have variable lengths → "Internal Fragmentation".
- **Result**: ~60% of memory wasted.

### 4.2 The Virtual Memory Analogy
PagedAttention treats KV cache like virtual memory in an OS:
1. Divide KV cache into **Physical Blocks**.
2. Each request has a **Logical Block Table**.
3. Blocks can be non-contiguous in HBM but appear contiguous to the kernel.

**Benefit**: Increases throughput by **2-4x** by fitting more requests into the same 80GB GPU.

---

## 5. Long Context Strategy (RoPE Scaling)

### 5.1 RoPE (Rotary Positional Embedding) Math
RoPE encodes relative position via rotation in complex space:
$$q_m = x_q e^{im\theta}, \quad k_n = x_k e^{in\theta}$$
$$\langle q_m, k_n \rangle = \text{Re}(x_q x_k^* e^{i(m-n)\theta})$$

### 5.2 Context Extension Techniques
How to go from 8K (training) to 128K (inference)?

#### 5.2.1 Linear Interpolation
$$f_{new}(pos) = f_{old}(pos \times \frac{L_{old}}{L_{new}})$$
**Issue**: Destroys high-frequency details.

#### 5.2.2 NTK-Aware Scaling (Dynamic Scaling)
Scale the **base frequency** $\theta$ instead of the position.
$$\theta_{new} = \theta \cdot \text{scale}$$
Allows the model to maintain precision for nearby tokens while expanding the global horizon.

---

## 6. Implementation: Flash-Attention Triton Kernel (Draft)

```python
@triton.jit
def flash_attn_kernel(
    Q, K, V, L,  # Pointers
    sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 1. Identify block coordinates
    m = tl.program_id(0)

    # 2. Load Q tile into SRAM
    q = tl.load(Q + m * BLOCK_M)

    # 3. Iterate over K, V blocks (The IO-aware loop)
    for n in range(0, N_CTX, BLOCK_N):
        k = tl.load(K + n * BLOCK_N)
        v = tl.load(V + n * BLOCK_N)

        # 4. Compute QK^T and apply Online Softmax
        qk = tl.dot(q, tl.trans(k))
        m_ij = tl.max(qk, 1)
        # ... update running stats and output ...
```

---

## 7. Comparative Performance (A100)

| Algorithm | Seq Len | Memory | Speed (ms/iter) |
|:--|:--|:--|:--|
| PyTorch Native | 8K | 16GB | 145 |
| FlashAttention-1 | 8K | 2GB | 42 |
| FlashAttention-2 | 8K | 2GB | **18** |
| FlashAttn-2 + GQA | 32K | 4GB | 55 |

---

## 8. Key References

1. **Dao et al. (2023)**: *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*.
2. **Kwon et al. (2023)**: *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM).
3. **Ainslie et al. (2023)**: *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*.
4. **Jiang et al. (2023)**: *Mistral 7B* (SWA introduction).
5. **Su et al. (2024)**: *RoFormer: Enhanced Transformer with Rotary Position Embedding*.

---

*Document follows the mathematical and systems engineering standards of NeurIPS/ICML 2024.*
