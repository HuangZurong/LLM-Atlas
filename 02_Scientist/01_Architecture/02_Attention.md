# Attention Mechanism

*Prerequisite: [01_Transformer.md](01_Transformer.md).*

This document derives Scaled Dot-Product Attention and Multi-Head Attention step by step, and covers modern variants (MQA, GQA, MLA, FlashAttention) used in today's LLMs.

## 1. Why Attention?

### 1.1 The Seq2Seq Bottleneck

In the classic Encoder-Decoder (Seq2Seq) architecture, the encoder compresses the entire input sequence into a single fixed-length context vector $\mathbf{c}$:

$$
\mathbf{c} = \text{Encoder}(x_1, x_2, \dots, x_n)
$$

The decoder relies solely on this one vector to generate all output tokens. The problems are clear:

- **Information bottleneck**: No matter how long the input is, all information is squeezed into a single fixed-dimension vector.
- **Long-range forgetting**: The longer the sequence, the more early-token information decays in $\mathbf{c}$.
- **Static compression**: Different output tokens need different source-side information, but $\mathbf{c}$ is the same for all of them.

### 1.2 The Core Idea of Attention

Attention is essentially a **dynamic weighted retrieval** mechanism:

> When generating each output token, let the model "look back" at all input tokens and assign different weights based on relevance.

In one sentence: **stop compressing — query on demand**.

---

## 2. From Bahdanau Attention to Self-Attention

### 2.1 Bahdanau Attention (Additive Attention, 2014)

Bahdanau et al. proposed that the decoder should have access to all encoder hidden states $(\mathbf{h}_1, \dots, \mathbf{h}_n)$ at every step, rather than relying only on the final state.

**Computation flow**:

1. **Alignment Score**:

$$
e_{ij} = \mathbf{v}^\top \tanh(\mathbf{W}_1 \mathbf{s}_{i-1} + \mathbf{W}_2 \mathbf{h}_j)
$$

where $\mathbf{s}_{i-1}$ is the decoder's previous hidden state and $\mathbf{h}_j$ is the encoder hidden state at position $j$.

2. **Attention Weights**:

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{n} \exp(e_{ik})}
$$

3. **Context Vector**:

$$
\mathbf{c}_i = \sum_{j=1}^{n} \alpha_{ij} \mathbf{h}_j
$$

Each decoding step produces a **different** $\mathbf{c}_i$, resolving the fixed-vector bottleneck.

### 2.2 Luong Attention (Multiplicative Attention, 2015)

Luong simplified the alignment function by replacing the additive network with a dot product:

$$
e_{ij} = \mathbf{s}_i^\top \mathbf{h}_j
$$

More efficient to compute, with comparable performance in practice. This laid the groundwork for Dot-Product Attention in the Transformer.

### 2.3 From Cross-Attention to Self-Attention

Both mechanisms above are **Cross-Attention**: the Query comes from the decoder, while Key/Value come from the encoder.

The key breakthrough of **Self-Attention** is that Query, Key, and Value all come from the **same sequence**. This allows every token to directly attend to every other token, without the sequential propagation required by RNNs.

## 3. Scaled Dot-Product Attention

This is the core computational unit of the Transformer.

### 3.1 The Query-Key-Value Abstraction

Attention can be framed as an **information retrieval** process:

| Concept | Analogy | Role |
| :--- | :--- | :--- |
| **Query** ($Q$) | Search keyword | "What am I looking for?" |
| **Key** ($K$) | Document title / index | "What information do I contain?" |
| **Value** ($V$) | Document content | "Here is my actual content" |

An input sequence $\mathbf{X} \in \mathbb{R}^{n \times d_{model}}$ is projected through three learnable matrices to produce Q, K, V:

$$
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
$$

where $W^Q, W^K \in \mathbb{R}^{d_{model} \times d_k}$ and $W^V \in \mathbb{R}^{d_{model} \times d_v}$.

### 3.2 The Formula

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

Step by step:

1. **Similarity computation**: $QK^\top \in \mathbb{R}^{n \times n}$ — the pairwise similarity matrix between all tokens.
2. **Scaling**: Divide by $\sqrt{d_k}$ to prevent large dot-product values from pushing softmax into saturation.
3. **Normalization**: Softmax converts each row into a probability distribution.
4. **Weighted aggregation**: Use the attention weights to compute a weighted sum of $V$.

### 3.3 Why Divide by $\sqrt{d_k}$?

Assume each element of $Q$ and $K$ is independently sampled from a distribution with mean 0 and variance 1. Then the dot product $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ has:

- **Mean** = 0
- **Variance** = $d_k$

When $d_k$ is large (e.g. 64 or 128), the dot-product magnitudes become large, softmax outputs approach a one-hot distribution, and gradients nearly vanish. Dividing by $\sqrt{d_k}$ normalizes the variance back to 1, keeping gradients flowing.

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch, seq_len, d_k)
    mask: (batch, 1, seq_len) or (batch, seq_len, seq_len)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights
```

### 3.4 Causal Mask

In Decoder-Only models, token $i$ can only attend to positions $\leq i$, preventing it from "seeing the future":

$$
\text{mask}_{ij} =
\begin{cases}
0 & \text{if } j > i \\
1 & \text{if } j \leq i
\end{cases}
$$

Masked positions are set to $-\infty$ before softmax, resulting in zero attention weight after normalization.

## 4. Multi-Head Attention (MHA)

### 4.1 Motivation

A single attention head can only learn one "attention pattern." But linguistic dependencies are multi-dimensional:

- Syntactic dependencies (subject-verb agreement)
- Semantic associations (synonyms, contextual meaning)
- Positional relationships (adjacent tokens)

Multi-Head Attention allows the model to learn multiple attention patterns **in parallel across different subspaces**.

### 4.2 Computation

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

where each head:

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

Parameter dimensions:

- Per-head projections: $W_i^Q, W_i^K \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$
- Typically $d_k = d_v = d_{model} / h$
- Output projection: $W^O \in \mathbb{R}^{hd_v \times d_{model}}$

**Total compute is the same as single-head attention**: although there are $h$ heads, each head's dimension is reduced to $d_{model}/h$.

### 4.3 Implementation

```python
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, N, _ = x.shape

        # Project and split into heads: (B, N, d_model) -> (B, h, N, d_k)
        Q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask)

        # Merge heads: (B, h, N, d_k) -> (B, N, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.W_o(attn_out)
```

## 5. Attention Variants in Modern LLMs

### 5.1 Multi-Query Attention (MQA)

**Problem**: In standard MHA, each head has its own K and V projections. During inference, the KV Cache consumes significant GPU memory.

**MQA** (Shazeer, 2019): All heads **share a single set** of K and V; only Q remains multi-headed.

$$
\text{KV Cache size}: \quad \text{MHA} = 2 \times h \times d_k \times L \quad \longrightarrow \quad \text{MQA} = 2 \times d_k \times L
$$

- KV Cache shrinks by a factor of $h$
- Significant inference speedup
- Slight quality degradation, but the gap is small in large models

### 5.2 Grouped-Query Attention (GQA)

GQA (Ainslie et al., 2023) is a **middle ground** between MHA and MQA. Instead of giving every query head its own KV pair (MHA) or forcing all heads to share one KV pair (MQA), GQA divides the $h$ query heads into $g$ groups, where each group shares one KV head.

$$
\text{MHA} \xrightarrow{g=h} \text{independent per head} \quad \longleftrightarrow \quad \text{MQA} \xrightarrow{g=1} \text{fully shared}
$$

#### Why GQA?

MQA's aggressive sharing can hurt quality on complex reasoning tasks. GQA recovers most of that quality while retaining the bulk of MQA's memory savings:

- With $g$ KV groups, KV Cache is $g/h$ the size of MHA — still a large reduction when $g \ll h$.
- Each KV head serves $h/g$ query heads, giving it a richer gradient signal than MQA's single KV head, which improves training stability.
- GQA models can be initialized from existing MHA checkpoints by **mean-pooling** the KV heads within each group, enabling efficient up-training rather than training from scratch.

#### Architecture Detail

Given $h$ query heads and $g$ KV groups (where $g$ divides $h$):

- Query projection: $W^Q \in \mathbb{R}^{d_{model} \times h \cdot d_k}$ — same as MHA.
- Key projection: $W^K \in \mathbb{R}^{d_{model} \times g \cdot d_k}$ — only $g$ heads instead of $h$.
- Value projection: $W^V \in \mathbb{R}^{d_{model} \times g \cdot d_k}$ — same as K.

During attention computation, each KV head is **broadcast** (repeated) to serve its corresponding $h/g$ query heads.

#### Comparison Table

| Variant | Query Heads | KV Heads | KV Cache | Representative Models |
| :--- | :---: | :---: | :---: | :--- |
| MHA | $h$ | $h$ | $2hd_kL$ | GPT-3, Llama 1 |
| GQA | $h$ | $g$ | $2gd_kL$ | Llama 2 70B, Llama 3, Mistral |
| MQA | $h$ | $1$ | $2d_kL$ | PaLM, Falcon |

#### Implementation

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        """
        n_heads:    number of query heads (h)
        n_kv_heads: number of KV heads (g), must divide n_heads
        """
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # how many Q heads per KV head
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(n_heads * self.d_k, d_model, bias=False)

    def forward(self, x, mask=None):
        B, N, _ = x.shape

        # Q: (B, N, h * d_k) -> (B, h, N, d_k)
        Q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        # K, V: (B, N, g * d_k) -> (B, g, N, d_k)
        K = self.W_k(x).view(B, N, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_kv_heads, self.d_k).transpose(1, 2)

        # Broadcast KV heads to match Q heads: (B, g, N, d_k) -> (B, h, N, d_k)
        if self.n_rep > 1:
            K = K.repeat_interleave(self.n_rep, dim=1)
            V = V.repeat_interleave(self.n_rep, dim=1)

        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.W_o(attn_out)
```

When `n_kv_heads == n_heads`, this reduces to standard MHA. When `n_kv_heads == 1`, it becomes MQA.

#### Practical Configurations

| Model | $h$ (Query) | $g$ (KV) | $h/g$ | $d_{model}$ |
| :--- | :---: | :---: | :---: | :---: |
| Llama 2 70B | 64 | 8 | 8 | 8192 |
| Llama 3 8B | 32 | 8 | 4 | 4096 |
| Llama 3 70B | 64 | 8 | 8 | 8192 |
| Mistral 7B | 32 | 8 | 4 | 4096 |

### 5.3 MLA (Multi-head Latent Attention)

Introduced by **DeepSeek-V2**, MLA is a revolutionary attention mechanism that goes beyond GQA to achieve even more extreme KV Cache compression while maintaining (or even exceeding) MHA performance.

**Core Idea**: Use **Low-Rank Joint Compression** to map K and V into a small "latent vector," which is then expanded back to multiple heads during computation.

1. **Compression**: $KV$ are compressed into a latent vector $c_{KV} \in \mathbb{R}^{d_{latent}}$ ($d_{latent} \ll h \cdot d_k$).
2. **Expansion**: During attention, $c_{KV}$ is expanded back to $h$ heads via a projection matrix.
3. **Decoupled RoPE**: To maintain RoPE compatibility (which doesn't commute with linear projection), MLA splits Query and Key into a compressed part (for content) and a separate part (for RoPE-based position encoding).

**Benefit**: DeepSeek-V2/V3 achieves KV Cache memory savings of **93.3%** compared to MHA, allowing for much larger batch sizes and context windows during inference.

### 5.4 FlashAttention

The bottleneck of standard attention is not compute but **memory access**. $QK^\top$ produces an $O(N^2)$ attention matrix that must be shuttled back and forth between HBM (high-bandwidth memory) and SRAM (on-chip cache).

**FlashAttention** (Dao et al., 2022) addresses this with three ideas:

1. **Tiling**: Split Q, K, V into small blocks that fit in SRAM.
2. **Online Softmax**: Incrementally compute softmax during the tiling pass — no need to store the full $N \times N$ matrix.
3. **Kernel Fusion**: Fuse matmul, scale, mask, softmax, and dropout into a single GPU kernel.

Results:

- Memory complexity drops from $O(N^2)$ to $O(N)$
- 2-4x wall-clock speedup (fewer HBM accesses)
- Numerically **exact** — not an approximation

### 5.5 Sliding Window Attention

Models like Mistral use sliding window attention: each token only attends to the most recent $w$ tokens (e.g. $w = 4096$).

- Computation drops from $O(N^2)$ to $O(Nw)$
- Through layer stacking, information can still propagate across the full sequence (receptive field grows linearly with depth)

## 6. Complexity Analysis

| Operation | Time Complexity | Space Complexity |
| :--- | :---: | :---: |
| $QK^\top$ computation | $O(N^2 d)$ | $O(N^2)$ |
| Softmax | $O(N^2)$ | $O(N^2)$ |
| Weighted sum | $O(N^2 d)$ | $O(Nd)$ |
| **Total (standard)** | $O(N^2 d)$ | $O(N^2 + Nd)$ |
| **FlashAttention** | $O(N^2 d)$ | $O(N)$ |
| **Sliding Window** | $O(Nwd)$ | $O(Nw)$ |

The $N^2$ computational complexity is the central challenge for long-sequence modeling, and the primary motivation behind alternative architectures like Linear Attention and Mamba (see [03_Efficient_Attention.md](03_Efficient_Attention.md)).

## 7. Summary

The evolution of attention:

$$
\text{Fixed Context Vector} \xrightarrow{\text{Bahdanau}} \text{Additive Attention} \xrightarrow{\text{Luong}} \text{Dot-Product Attention} \xrightarrow{\text{Vaswani}} \text{Scaled Dot-Product + Multi-Head}
$$

Core formula:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

Key design choices:

- **Scaling**: $\sqrt{d_k}$ prevents gradient vanishing in softmax
- **Multi-Head**: Parallel attention patterns across subspaces, no extra compute
- **KV sharing** (MQA/GQA): Trade-off between inference efficiency and quality
- **IO-aware** (FlashAttention): Same algorithm, optimized memory access patterns

## 8. Key References

1. **Bahdanau et al. (2014)**: *Neural Machine Translation by Jointly Learning to Align and Translate*.
2. **Luong et al. (2015)**: *Effective Approaches to Attention-based Neural Machine Translation*.
3. **Vaswani et al. (2017)**: *Attention Is All You Need*.
4. **Shazeer (2019)**: *Fast Transformer Decoding: One Write-Head is All You Need* (MQA).
5. **Dao et al. (2022)**: *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*.
6. **Ainslie et al. (2023)**: *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*.
7. **DeepSeek-AI (2024)**: *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model* (MLA).
