# Position Encoding

*Prerequisite: [01_Transformer.md](01_Transformer.md), [02_Attention.md](02_Attention.md). For long-context RoPE scaling (PI, NTK, YaRN, ABF), see [12_Long_Context.md](12_Long_Context.md).*

---

## 1. Why Position Encoding?

Self-attention is **permutation-invariant** — it treats the input as a set, not a sequence. Without positional information, "The cat sat on the mat" and "The mat sat on the cat" produce identical attention outputs.

Position Encoding (PE) injects order information so the model can distinguish token positions.

### 1.1 Original Approach: Sinusoidal / Learned Absolute PE

The original Transformer ("Attention is All You Need") adds a position vector directly to the token embedding:

$$
\text{input} = \text{token\_embedding} + \text{position\_embedding}
$$

Problems:

- Encodes **absolute position** — poor generalization when sequence length exceeds training length.
- Position and semantic information are **mixed additively**, which can cause overfitting.
- The model has no explicit mechanism to capture **relative distance** between tokens.

## 2. Core Intuition of RoPE

Three guiding principles for a good positional encoding:

1. **Closer tokens should attend more strongly** — position encoding should create a natural distance-based decay.
2. **Position-agnostic** — the encoding should not depend on the total sequence length.
3. **Relative over absolute** — what matters is the **distance between two tokens** $(m - n)$, not their absolute positions $m$ and $n$.

RoPE achieves all three by encoding position as a **rotation** applied to Q and K vectors.

## 3. Mathematical Derivation

### 3.1 The Constraint

We want a transformation $f$ such that the dot product of two transformed vectors depends **only on their content and relative distance**:

$$
\langle f(\mathbf{q}, m),\ f(\mathbf{k}, n) \rangle = g(\mathbf{q}, \mathbf{k}, m - n)
$$

where $m$ and $n$ are absolute positions, but the result depends only on $(m - n)$.

### 3.2 Complex Number Representation

**Key insight**: pair up adjacent embedding dimensions and treat each pair as a **complex number**.

A $d$-dimensional real vector in $\mathbb{R}^d$ becomes a $d/2$-dimensional complex vector in $\mathbb{C}^{d/2}$:

$$
[x_0, x_1, x_2, x_3, \ldots] \rightarrow [(x_0 + ix_1),\ (x_2 + ix_3),\ \ldots]
$$

### 3.3 Rotation as Position Encoding

For each complex pair at dimension index $j$, apply a rotation by angle $m \cdot \theta_j$:

$$
\text{RoPE}(\mathbf{x}, m)_j = x_j \cdot e^{im\theta_j}
$$

Here $i$ is the imaginary unit ($\sqrt{-1}$), $m$ is the token position, and $j$ is the dimension pair index. The frequency for each pair is:

$$
\theta_j = \frac{1}{10000^{2j/d}}
$$

This is the same frequency schedule as the original sinusoidal PE, but used **multiplicatively** instead of additively.

### 3.4 Why This Works

Compute the inner product of two RoPE-encoded vectors:

$$
\langle \text{RoPE}(\mathbf{q}, m),\ \text{RoPE}(\mathbf{k}, n) \rangle = \sum_j q_j \cdot k_j^* \cdot e^{im\theta_j} \cdot e^{-in\theta_j} = \sum_j q_j \cdot k_j^* \cdot e^{i(m-n)\theta_j}
$$

Writing in vector form with $R(\theta)$ denoting the rotation operator:

$$
\langle \text{RoPE}(\mathbf{q}, m),\ \text{RoPE}(\mathbf{k}, n) \rangle = \mathbf{q}^\top \cdot R(-m\theta) \cdot R(n\theta) \cdot \mathbf{k} = \mathbf{q}^\top \cdot R((n - m)\theta) \cdot \mathbf{k}
$$

This makes it clear: the rotation $R$ depends **only on the relative distance $(n-m)$**, not on $m$ or $n$ individually. The absolute positions have been completely eliminated.

### 3.5 Rotation Matrix Form

Expanding the complex multiplication into real-valued $2 \times 2$ rotation matrices:

For each dimension pair $(2j, 2j+1)$:

$$
\begin{pmatrix} \cos(m\theta_j) & -\sin(m\theta_j) \\ \sin(m\theta_j) & \cos(m\theta_j) \end{pmatrix} \begin{pmatrix} x_{2j} \\ x_{2j+1} \end{pmatrix}
$$

The full rotation matrix is block-diagonal — each $2 \times 2$ block rotates one dimension pair independently, with its own frequency $\theta_j$.

## 4. Implementation

### 4.1 Efficient Computation

Instead of constructing the full rotation matrix, use the identity:

```python
def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

where `rotate_half` swaps and negates adjacent pairs:

$$
\text{rotate\_half}([x_0, x_1, x_2, x_3]) = [-x_1, x_0, -x_3, x_2]
$$

This is equivalent to the $2 \times 2$ rotation matrix multiplication but avoids materializing the matrix.

### 4.2 Important Detail: Only Q and K

**RoPE is applied only to Q and K, not to V.**

Reason: positional information only needs to influence the **attention score** ($QK^\top$ dot product). The value aggregation (weighted sum of V) does not need position-dependent rotation.

### 4.3 Precomputation

The $\cos$ and $\sin$ tables can be **precomputed** for all positions up to `max_seq_len` and cached, since they depend only on position and dimension index, not on input content.

```python
# Precompute frequency table
freqs = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))

# Precompute for all positions
t = torch.arange(max_seq_len)
freqs = torch.outer(t, freqs)          # (max_seq_len, d/2)
cos_cached = freqs.cos()                # (max_seq_len, d/2)
sin_cached = freqs.sin()                # (max_seq_len, d/2)
```

## 5. Comparison with Other PE Methods

| Property                         | Sinusoidal  | Learned Absolute | T5 Relative Bias           | **RoPE**                      |
| -------------------------------- | ----------- | ---------------- | -------------------------- | ----------------------------------- |
| How applied                      | Additive    | Additive         | Additive bias to attention | **Multiplicative (rotation)** |
| Position type                    | Absolute    | Absolute         | Relative                   | **Relative**                  |
| Dimension coupling               | Independent | Independent      | N/A                        | **Paired (2D rotation)**      |
| Length generalization            | Poor        | Poor             | Moderate                   | **Good**                      |
| Compatible with linear attention | Yes         | Yes              | No                         | **Yes**                       |
| Extra parameters                 | None        | $d \times L$   | $O(L)$ bias              | **None**                      |

## 6. Experimental Results

On language modeling benchmarks (EleutherAI):

| Model Size  | Learned Absolute | T5 RPE | **RoPE**  |
| ----------- | ---------------- | ------ | --------------- |
| 125M params | 2.809            | 2.801  | **2.759** |
| 1.4B params | 2.240            | 2.223  | **2.173** |

- ~30% faster convergence vs. learned absolute
- 10-20% improvement over T5 relative encoding
- Negligible runtime overhead (1-3% with kernel fusion)

## 7. Summary

RoPE encodes position by **rotating** Q and K vectors in 2D subspaces. The rotation angle is proportional to the token's position, and different dimension pairs rotate at different frequencies. This elegantly ensures that the attention dot product depends only on relative distance, with zero extra parameters and minimal computational overhead.

$$
\text{Core formula:}\quad \text{RoPE}(\mathbf{x}, m) = \mathbf{x} \cdot e^{im\theta}
$$

$$
\text{Key property:}\quad \langle \text{RoPE}(\mathbf{q}, m),\ \text{RoPE}(\mathbf{k}, n) \rangle \text{ depends only on } (m - n)
$$

## 8. Key References

1. **Vaswani et al. (2017)**: *Attention Is All You Need* (Sinusoidal PE).
2. **Shaw et al. (2018)**: *Self-Attention with Relative Position Representations*.
3. **Su et al. (2021)**: *RoFormer: Enhanced Transformer with Rotary Position Embedding* (RoPE).
4. **Raffel et al. (2020)**: *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5 Relative Bias).
5. **Press et al. (2022)**: *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation* (ALiBi).
