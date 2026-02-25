# Dense Transformer Architecture

*Prerequisite: [01_Transformer.md](01_Transformer.md), [02_Attention.md](02_Attention.md).*

---

## 1. Overview

The Dense Transformer is the standard architecture where every parameter participates in every forward pass. It remains the foundational building block upon which MoE and Hybrid variants are constructed.

**Core formula per layer:**

$$\text{Output} = \text{FFN}(\text{MultiHeadAttn}(x) + x) + \text{MultiHeadAttn}(x) + x$$

## 2. Layer Structure

A standard Dense Transformer block consists of:

```
Input
  → LayerNorm
  → Multi-Head Attention + Residual
  → LayerNorm
  → FFN (Feed-Forward Network) + Residual
Output
```

### 2.1 Pre-Norm vs Post-Norm

- **Post-Norm (Original)**: Normalization after the residual addition. Used in the original Transformer and BERT.
- **Pre-Norm (Modern)**: Normalization before the sub-layer. Adopted by GPT-2, Llama, and most modern LLMs for better training stability at scale.

### 2.2 Normalization Variants

| Method | Formula | Used By |
|--------|---------|---------|
| LayerNorm | $\frac{x - \mu}{\sigma} \cdot \gamma + \beta$ | GPT, BERT |
| RMSNorm | $\frac{x}{\text{RMS}(x)} \cdot \gamma$ | Llama, Mistral |

RMSNorm removes the mean-centering step, reducing compute while maintaining performance.

## 3. FFN: The Parameter-Heavy Component

In a Dense Transformer, the FFN layer accounts for ~2/3 of total parameters per block. This is exactly the component that MoE replaces with multiple experts.

### 3.1 Standard FFN

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

Two matrices: $W_1$ (up-projection) and $W_2$ (down-projection).

### 3.2 SwiGLU Activation Function

In the evolution of deep learning, activation functions play the role of neuronal "switches." Early ReLU was simple and effective, solving the vanishing gradient problem; GeLU introduced probabilistic thinking in BERT and GPT-2; and in the era dominated by Llama, Mistral, and MoE, SwiGLU has become the new architectural standard due to its superior performance.

#### 3.2.1 Core Intuition: From "Single Lane" to "Dual Lane"

To understand SwiGLU's superiority, first understand how it changes information processing.

- **Traditional Mode (ReLU/GeLU)**: **"Single Lane + Toll Booth"**

  Traditional FFN has only one pathway. The input signal is amplified, then encounters the activation function (toll booth).

  - **Logic**: If positive, pass through; if negative, block (set to 0).
  - **Flaw**: This "hard cutoff" causes permanent information loss (neuron death) and cannot flexibly adjust the pass-through ratio based on context.

- **SwiGLU Mode**: **"Dual Lane + Smart Valve"**

  SwiGLU introduces the GLU (Gated Linear Unit) mechanism, duplicating the input signal into two parallel paths:

  1. **Value Path**: Carries the actual information content.
  2. **Gate Path**: Computes a "valve opening" between 0 and 1.

  The two paths merge, using the gate path's coefficient to modulate the value path's information. This means the model can learn **"for this input, should I retain 10% or 90% of this feature?"** instead of a rigid binary cutoff.

#### 3.2.2 Mathematical Definition & Structural Decomposition

SwiGLU stands for **Swish-Gated Linear Unit**, combining the Swish activation function with the GLU gating structure.

##### 1. Base Component: Swish Activation Function

The "gate" in SwiGLU is not a simple Sigmoid, but the Swish function (typically with $\beta=1$, i.e., SiLU):

$$\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

- **Properties**: A smooth curve. Unlike ReLU, it allows small negative values on the negative half-axis (no dead zone), and is smooth and differentiable everywhere, enabling extremely smooth gradient propagation in deep networks.

##### 2. SwiGLU FFN Formula

In the Transformer's feed-forward layer, SwiGLU computes as follows:

$$\text{FFN}_{\text{SwiGLU}}(x) = (\underbrace{\text{Swish}(x W_g)}_{\text{gate signal}} \odot \underbrace{(x W_u)}_{\text{content}}) W_d$$

- $x$: Input vector.
- $W_g$ (Gate): Gate projection matrix, computes the "pass-through rate."
- $W_u$ (Up): Up-projection matrix, transforms the "content."
- $\odot$: Element-wise multiplication (Hadamard Product), combining "gate" and "content."
- $W_d$ (Down): Down-projection matrix, maps the result back to the original dimension.

#### 3.2.3 Physical Meaning & Performance Advantages

Why is this complex structure better than simple ReLU?

1. **Stronger Nonlinear Expressiveness (Quadratic Interaction)**

   Note the $(x W_g) \odot (x W_u)$ in the formula. This means input $x$ is multiplied with itself. Mathematically, this introduces **$x^2$ (quadratic)** level higher-order feature interactions. Compared to ReLU's simple linear transformation and truncation, SwiGLU captures more complex relationships between features.

2. **Gradient Stability**

   The Swish function itself contains $x$ and the Sigmoid derivative term, and is smooth everywhere. This ensures that during backpropagation through hundreds of layers, gradient signals remain clear, resistant to explosion or vanishing.

#### 3.2.4 Engineering Tradeoff: The Origin of the "2/3" Coefficient

SwiGLU performs well but has an obvious "drawback": **it uses one extra matrix**.

- **Standard FFN**: 2 matrices ($W_{up}, W_{down}$).
- **SwiGLU**: 3 matrices ($W_{gate}, W_{up}, W_{down}$).

Using SwiGLU directly at the same hidden width would increase parameters and compute by 50%. To maintain a **fair comparison under the same parameter budget**, we need to reduce the hidden width.

##### Derivation:

Given input dimension $d$ and hidden width $h$:

1. **Standard FFN parameters** $\approx 2 \times d \times h$
2. **SwiGLU parameters** $\approx 3 \times d \times h'$ (let new width be $h'$)

To match total parameters (Cost Match):

$$3 \times d \times h' = 2 \times d \times h$$

$$h' = \frac{2}{3} h$$

##### Conclusion:

To offset the extra matrix cost, SwiGLU's hidden width is typically set to **$2/3$** of the standard width.

In standard Transformers, hidden width $h$ is usually $4d$. Thus in Llama and similar models, SwiGLU width becomes:

$$d_{ff} \approx \frac{2}{3} \times 4d = \frac{8}{3}d$$

This is why Llama 2 (7B) has an intermediate dimension of **11008** instead of the standard 16384 — because $11008 \approx \frac{2}{3} \times 16384$. A classic "trade width for depth (complex interaction)" engineering decision.

### 3.3 Deep Analysis of Projection Layers

The three projection matrices play specific roles:

- **Gate Projection ($W_g$) & Up Projection ($W_u$)**:
  - These two matrices map the input token vector from model dimension ($d_{model}$) to a higher-dimensional intermediate feature space ($d_{ff}$).
  - **Up Projection** provides rich information content (Value).
  - **Gate Projection** provides the control signal for selecting information (Attention/Gating).
  - This separation allows the model to independently learn "content" and "control," similar to LSTM gating logic but implemented in parallel within the feed-forward network.
- **Down Projection ($W_d$)**:
  - Maps the high-dimensional intermediate features back to model dimension ($d_{model}$).
  - This is not merely dimensionality reduction, but feature aggregation. After gating and nonlinear transformation, features are linearly combined here to form the final output for the token.
  - Quantization research has found that **Down Projection** is extremely sensitive to numerical precision and typically cannot be aggressively quantized, while **Up Projection** is relatively robust.

### 3.4 SwiGLU Implementation Example

```python
class FeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network

    Implements a feed-forward network with SwiGLU (Swish-Gated Linear Unit) activation.
    SwiGLU is a GLU variant that uses Swish/SiLU as the gating activation function.

    Formula:
        FFN(x) = down_proj(Swish(gate_proj(x)) * up_proj(x))

    Where:
        - gate_proj: Gate projection, generates the gating signal
        - up_proj: Up projection, generates features
        - Swish(x) = x * sigmoid(x) = x * silu(x)
        - down_proj: Down projection, maps intermediate dimension back to hidden_size

    Compared to standard FFN (ReLU(xW1)W2), SwiGLU typically achieves better performance.
    """
    def __init__(self, config: ModelConfig):
        """
        Initialize the feed-forward network

        Args:
            config: Model configuration object
        """
        super().__init__()
        # ========== Intermediate dimension calculation ==========
        # If intermediate_size is not specified, compute automatically
        if config.intermediate_size is None:
            # Standard ratio: intermediate_size = hidden_size * 8/3
            #   e.g.: hidden_size=512 -> intermediate_size ≈ 1365
            intermediate_size = int(config.hidden_size * 8 / 3)
            # Round up to nearest multiple of 64 (optimizes GPU compute efficiency)
            #   e.g.: 1365 -> 1408 (64 * 22)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        # ========== Projection layers ==========
        # gate_proj: Gate projection, hidden_size -> intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # down_proj: Down projection, intermediate_size -> hidden_size
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        # up_proj: Up projection, hidden_size -> intermediate_size
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

        # ========== Dropout and activation function ==========
        self.dropout = nn.Dropout(config.dropout)
        # Activation function: typically 'silu' (Swish)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """
        Forward pass

        SwiGLU formula: FFN(x) = down_proj(Swish(gate_proj(x)) * up_proj(x))

        Args:
            x: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        # Compute gate signal and features
        gate = self.gate_proj(x)  # [batch, seq_len, intermediate_size]
        up = self.up_proj(x)  # [batch, seq_len, intermediate_size]

        # SwiGLU: Swish(gate) * up
        #   Swish(x) = x * sigmoid(x) = silu(x)
        activated = self.act_fn(gate) * up  # [batch, seq_len, intermediate_size]

        # Down-project back to hidden_size and apply dropout
        return self.dropout(self.down_proj(activated))  # [batch, seq_len, hidden_size]
```

## 4. Scaling Properties

Dense models scale predictably but expensively:

| Model | Params | Layers | Hidden Dim | FFN Dim |
|-------|--------|--------|-----------|---------|
| GPT-3 | 175B | 96 | 12288 | 49152 |
| Llama 2 7B | 7B | 32 | 4096 | 11008 |
| Llama 2 70B | 70B | 80 | 8192 | 28672 |
| Llama 3 405B | 405B | 126 | 16384 | 53248 |

**Key constraint**: FLOPs per token ∝ Total Parameters. Doubling the model size doubles the inference cost — there is no way around this in a Dense architecture.

## 5. Strengths & Limitations

**Strengths:**
- Training stability — no load balancing or routing issues
- Uniform processing — every token gets the full model capacity
- Simple distributed training — standard tensor/pipeline parallelism
- Well-understood scaling laws (Chinchilla, Kaplan et al.)

**Limitations:**
- Compute wall — inference cost grows linearly with parameters
- Redundancy — not all parameters are equally useful for every token
- The FFN layer is the bottleneck, both in parameters and FLOPs

## 6. From Dense to MoE: The Transition Point

The Dense FFN is the natural insertion point for MoE. By replacing a single large FFN with N smaller expert FFNs + a router, we break the linear coupling between parameters and compute:

```
Dense:   Token → [Single FFN (all params)] → Output
MoE:     Token → Router → [Top-K Expert FFNs (subset of params)] → Weighted Sum → Output
```

This architectural relationship is why understanding the Dense block thoroughly is a prerequisite for understanding MoE — every expert inside an MoE layer is essentially a smaller Dense FFN.

For the full MoE deep dive, see [09_MoE_Arch.md](09_MoE_Arch.md).

---

## 7. Key References

1. **Vaswani et al. (2017)**: *Attention Is All You Need*.
2. **Shazeer (2020)**: *GLU Variants Improve Transformer* (SwiGLU).
3. **Touvron et al. (2023)**: *LLaMA: Open and Efficient Foundation Language Models*.
4. **Dubey et al. (2024)**: *The Llama 3 Herd of Models*.
