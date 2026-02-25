# Transformer Architecture

*Prerequisite: None (Foundational).*
*See Also: [../../01_Fundamentals/04_Transformer_Era/02_Transformer.md](../../01_Fundamentals/04_Transformer_Era/02_Transformer.md) (introductory level), [../../03_Engineering/01_LLMs/01_Theory/01_Intelligence_Landscape.md](../../03_Engineering/01_LLMs/01_Theory/01_Intelligence_Landscape.md) (engineering application).*

---

## 1. The Evolution of Language Modeling

The journey from static word vectors to massive autoregressive scaling:

1. **Word2Vec (2013)**: Static word-level embeddings via self-supervised learning.
2. **RNN/LSTM (2014–2016)**: Sequential processing with hidden states; limited by vanishing gradients and poor parallelization.
3. **ELMo (2018)**: Contextualized embeddings using deep Bi-LSTMs.
4. **BERT (2018)**: Bidirectional encoding via Transformer blocks; revolutionized NLU.
5. **GPT (2018–Present)**: Unidirectional generation via Decoder-only Transformers; established the scaling paradigm.

---

## 2. Core Architecture Components

The original Transformer (Vaswani et al., 2017) introduced the Encoder-Decoder structure. Modern LLMs have evolved these components for extreme scale.

### 2.1 Multi-Head Attention (MHA)

Allows the model to jointly attend to information from different representation subspaces. See [02_Attention.md](02_Attention.md) for full derivation.

### 2.2 Feed-Forward Networks (FFN)

Applied to each position separately and identically. In modern models (e.g., Llama), the standard ReLU FFN is replaced by **SwiGLU** for better performance.

### 2.3 Layer Normalization

| Variant | Description |
|:--|:--|
| **Post-Norm** | Original Transformer default. Better final performance but harder to train. |
| **Pre-Norm** | Modern LLM standard (GPT-3, Llama). More stable training at scale. |
| **RMSNorm** | Simplified variant used in Llama/DeepSeek; removes mean-centering for speed. |

### 2.4 Positional Encoding

Since Transformers process all tokens in parallel, they lack inherent sequence order.

- **Sinusoidal**: Original absolute encoding (Vaswani et al., 2017).
- **RoPE**: The modern standard; encodes relative position via rotation matrices. See [06_Position_Encoding.md](06_Position_Encoding.md).

---

## 3. Paradigm Convergence: Decoder-Only

The industry has converged on **Decoder-Only** architectures for generative LLMs.

### 3.1 Why Decoder-Only?

- **Unified Objective**: Next-token prediction is sufficient for both understanding and generation.
- **Scaling Efficiency**: Simpler to optimize KV caching and parallelize across massive clusters.
- **In-Context Learning**: Decoder models exhibit stronger few-shot capabilities at scale.

### 3.2 Key Mechanisms

- **Causal Masking**: Token $i$ can only attend to positions $\leq i$, preventing information leakage from future tokens.
- **KV Cache**: Stores previously computed Key/Value vectors to avoid $O(N^2)$ recomputation during auto-regressive decoding. KV Cache size is the primary memory bottleneck for long-context models.

---

## 4. Architectural Evaluation Dimensions

| Dimension | Description |
|:--|:--|
| **Parameter Efficiency** | Performance per parameter (e.g., Llama 3 8B vs. larger models) |
| **Inference Latency** | Throughput (tokens/sec) and Time To First Token (TTFT) |
| **Context Window** | Maximum effective sequence length (RoPE scaling, see [12_Long_Context.md](12_Long_Context.md)) |
| **KV Cache Efficiency** | Memory per token during inference (MHA vs. GQA vs. MLA) |

---

## 5. Key References

1. **Vaswani et al. (2017)**: *Attention Is All You Need*.
2. **Radford et al. (2018)**: *Improving Language Understanding by Generative Pre-Training* (GPT-1).
3. **Zhang & Sennrich (2019)**: *Root Mean Square Layer Normalization* (RMSNorm).
4. **Touvron et al. (2023)**: *LLaMA: Open and Efficient Foundation Language Models*.
