# Long Context: Scaling Transformers Beyond Training Length

*Prerequisite: [06_Position_Encoding.md](06_Position_Encoding.md) (RoPE fundamentals), [02_Attention.md](02_Attention.md) (attention mechanics).*

---

## 1. The Three Bottlenecks

| Bottleneck | Root Cause | Scaling |
|:--|:--|:--|
| **Attention compute** | Self-attention is $O(N^2 d)$ | 4× cost when context doubles |
| **KV Cache memory** | Each token stores K, V vectors for all layers | Linear in $N$, but dominates GPU memory at long contexts |
| **Position extrapolation** | RoPE trained on length $L$ degrades at $L' > L$ | Attention scores become numerically unstable at unseen positions |

A complete long-context solution must address all three simultaneously.

---

## 2. RoPE Scaling: Extending Position Encoding

All methods below modify RoPE's frequency schedule $\theta_j = 1/10000^{2j/d}$ to support longer contexts without full retraining.

### 2.1 Position Interpolation (PI)

**Chen et al. (2023)** — *Extending Context Window of Large Language Models via Positional Interpolation*

Instead of extrapolating to unseen positions, **compress** all positions to fit within the original training range:

$$\theta_j^{PI} = \frac{\theta_j}{s}, \quad s = \frac{L'}{L}$$

where $s$ is the scaling factor (e.g., $s=4$ to extend 4K → 16K).

- **Pros**: Simple, stable, works with minimal fine-tuning (~1000 steps).
- **Cons**: Compresses *all* frequencies equally — high-frequency dimensions (which encode local patterns) lose resolution.

### 2.2 NTK-aware Interpolation

**bloc97 (2023)** — *NTK-Aware Scaled RoPE*

Key insight: high-frequency RoPE dimensions encode **local** position (nearby tokens), while low-frequency dimensions encode **global** position (distant tokens). Only the global dimensions need scaling.

Modify the base frequency:

$$\theta_j^{NTK} = \frac{1}{(b \cdot s^{d/(d-2)})^{2j/d}}$$

where $b = 10000$ is the original base. This effectively:
- Leaves high-frequency dimensions (small $j$) nearly unchanged.
- Scales low-frequency dimensions (large $j$) more aggressively.

**Dynamic NTK**: Adjust $s$ at runtime based on actual sequence length — no fixed scaling factor needed.

### 2.3 YaRN (Yet another RoPE extensioN)

**Peng et al. (2023)** — arXiv:2309.00071

Combines NTK-aware interpolation with two additional techniques:

1. **Frequency-dependent interpolation**: Partition RoPE dimensions into three groups:
   - High-frequency → no interpolation (extrapolate)
   - Mid-frequency → partial interpolation (blend)
   - Low-frequency → full interpolation

   Controlled by a ramp function between wavelength thresholds $\alpha$ and $\beta$.

2. **Attention scaling**: Multiply attention logits by a temperature factor $\sqrt{t}$ to counteract the entropy increase from longer sequences:
   $$\text{Attention}(Q, K) = \frac{QK^\top}{\sqrt{d}} \cdot \frac{1}{\sqrt{t}}$$

- **Key Result**: Extends Llama 2 from 4K → 128K with only ~400 fine-tuning steps.

### 2.4 ABF (Adjusted Base Frequency) — Llama 3.1

**Meta (2024)** — *The Llama 3 Herd of Models*

The simplest effective approach: just increase the RoPE base frequency:

$$\theta_j^{ABF} = \frac{1}{b_{new}^{2j/d}}, \quad b_{new} = 500000$$

(Original base $b = 10000$, Llama 3.1 uses $b_{new} = 500000$.)

**Why it works**: A larger base stretches all wavelengths, making the model see longer sequences as "shorter" in rotation space. Combined with continued pre-training on long documents, this achieves 128K context.

### 2.5 Comparison

| Method | Mechanism | Fine-tuning | Context Achieved |
|:--|:--|:--|:--|
| **PI** | Uniform compression | ~1000 steps | 4K → 32K |
| **NTK-aware** | Frequency-dependent base scaling | Minimal / zero-shot | 4K → 16K |
| **YaRN** | NTK + ramp interpolation + attn scaling | ~400 steps | 4K → 128K |
| **ABF** | Increase base frequency | Continued pre-training | 8K → 128K |

**Current practice (2024–2025)**: Most frontier models use ABF or YaRN-style approaches combined with continued pre-training on long-context data. The trend is toward training-time solutions rather than post-hoc fixes.

---

## 3. Attention-Level Solutions

### 3.1 Ring Attention

**Liu et al. (2023)** — arXiv:2310.01889

Distributes the KV sequence across multiple devices in a ring topology:

1. Each device holds one **block** of the KV sequence.
2. Q blocks are computed locally; KV blocks are passed around the ring.
3. Each device computes partial attention with the current KV block, then passes it to the next device.

**Result**: Context length scales linearly with device count. Complementary to FlashAttention (FA optimizes single-device IO; Ring Attention optimizes cross-device distribution).

### 3.2 Sparse Attention Patterns

| Pattern | Description | Example |
|:--|:--|:--|
| **Sliding window** | Each token attends only to the nearest $w$ tokens | Mistral (w=4096) |
| **Global + local** | Designated tokens attend to everything; others attend locally | Longformer, BigBird |
| **Learned sparse (NSA)** | Model learns which tokens to attend to, aligned to hardware blocks | DeepSeek NSA (2025) |

Mistral's **sliding window attention** is notable for its simplicity: stack $L$ layers with window $w$, and the effective receptive field is $L \times w$ — information propagates through layers without any token attending beyond $w$.

### 3.3 Context Parallelism

Used in training (not inference): split the sequence dimension across devices, each computing attention on its chunk. Requires all-to-all communication for cross-chunk attention. Implementations: Megatron-LM sequence parallelism, DeepSpeed Ulysses.

---

## 4. KV Cache Management

At inference time, the KV Cache is the primary memory bottleneck. For a 70B model at 128K context, KV Cache alone can exceed 40 GB.

### 4.1 Sliding Window / StreamingLLM

**Xiao et al. (2023)** — *Efficient Streaming Language Models with Attention Sinks*

Discovery: the first few tokens ("attention sinks") receive disproportionately high attention regardless of content. Keeping these sink tokens + a sliding window of recent tokens enables **infinite-length** streaming inference:

$$\text{KV Cache} = [\text{sink tokens}] \cup [\text{recent } w \text{ tokens}]$$

- Fixed memory regardless of total sequence length.
- Trade-off: tokens outside the window are permanently lost.

### 4.2 KV Cache Compression

| Method | Approach |
|:--|:--|
| **GQA / MQA** | Reduce number of KV heads at architecture level (see [02_Attention.md](02_Attention.md)) |
| **MLA** | Low-rank joint compression of KV (see [02_Attention.md](02_Attention.md) §5.3) |
| **Quantized KV** | Quantize cached K, V to INT4/INT8 (e.g., KIVI, KVQuant) |
| **H2O** | Heavy-Hitter Oracle — evict low-attention KV entries dynamically |
| **Token merging** | Merge similar KV entries to reduce cache size |

**MLA** (DeepSeek-V2) achieves 93.3% KV Cache reduction — the most aggressive architectural solution to date.

---

## 5. Training on Long Contexts

Simply extending position encoding is insufficient — the model must also **learn** to use long-range information.

### 5.1 Progressive Training

Standard approach used by Llama 3.1, Qwen2, and others:

1. **Phase 1**: Pre-train on short contexts (4K–8K) for the majority of tokens.
2. **Phase 2**: Continue pre-training on long documents (32K–128K) with adjusted RoPE, using a smaller learning rate.
3. **Phase 3**: Long-context SFT with instruction-following data at target length.

**Rationale**: Short-context training is more compute-efficient (more tokens per batch). Long-context training is a fine-tuning phase, not the main pre-training.

### 5.2 Data Requirements

Long-context training requires documents that genuinely contain long-range dependencies:
- Books, legal documents, codebases, multi-turn conversations
- Synthetic tasks: multi-document QA, summarization of concatenated articles
- **Not** just padding short documents — the model must learn to attend across the full range.

---

## 6. Evaluation

### 6.1 Needle In A Haystack (NIAH)

Insert a target fact at various positions within a long document and test retrieval. Simple but widely used.

**Limitation**: Only tests verbatim retrieval, not reasoning over long context.

### 6.2 RULER (Hsieh et al., 2024)

Extends NIAH with harder tasks:
- Multi-key NIAH (retrieve multiple needles)
- Multi-value NIAH (one key maps to multiple values)
- Variable tracking across long context
- Common/frequent word extraction

**Key finding**: Many models claiming 128K+ context show significant degradation beyond 32K on RULER tasks.

### 6.3 Practical Observations

- **Claimed vs. effective context**: Most models perform well up to ~32K but degrade significantly at their claimed maximum. Gemini 1.5 Pro is a notable exception with >99% NIAH recall at 1M tokens.
- **Lost in the middle**: Models tend to attend more to the beginning and end of context, underweighting information in the middle (Liu et al., 2023).

---

## 7. Key References

1. **Chen et al. (2023)**: *Extending Context Window of Large Language Models via Positional Interpolation*.
2. **Peng et al. (2023)**: *YaRN: Efficient Context Window Extension of Large Language Models* — arXiv:2309.00071.
3. **Liu et al. (2023)**: *Ring Attention with Blockwise Transformers for Near-Infinite Context* — arXiv:2310.01889.
4. **Xiao et al. (2023)**: *Efficient Streaming Language Models with Attention Sinks* (StreamingLLM).
5. **Meta (2024)**: *The Llama 3 Herd of Models* — ABF scaling to 128K context.
6. **Hsieh et al. (2024)**: *RULER: What's the Real Context Size of Your Long-Context Language Models?*
7. **Liu et al. (2023)**: *Lost in the Middle: How Language Models Use Long Contexts*.
