# ModernBERT Fine-Tuning (Encoder-only NLU)

*Prerequisite: [../../01_Theory/01_Introduction.md](../../01_Theory/01_Introduction.md).*

---

ModernBERT is a modernization of BERT maintaining full backward compatibility while delivering dramatic improvements through architectural innovations like **Rotary Positional Embeddings (RoPE)**, **Alternating Attention patterns**, and hardware-optimized design. 

It was trained on **2 trillion tokens** of diverse data (web, code, scientific articles), making it significantly more robust than traditional BERT models.

### Key Specifications:
- **Context Length**: Native support for **8192 tokens** (approx. 16x larger than vanilla BERT).
- **Efficiency**: 2-4x faster than previous encoder models (e.g., DeBERTa-v3).
- **Models**: Available in **Base (139M)** and **Large (395M)** sizes.
- **Hardware-Aware**: Optimized for **FlashAttention-2** and unpadding out of the box.

## 1. Core Objectives: From NLU Basics to Business Logic

ModernBERT supports both standard NLU tasks and high-precision industrial logic:

- **Standard NLU Tasks**:
  - **Sequence Classification**: Sentiment analysis, topic categorization.
  - **Token Classification**: NER, PII masking.
  - **Semantic Similarity**: Retrieval, re-ranking, embedding alignment.
- **Architectural Logic Inference**:
  - **Long-Context Logic**: Utilizing the 8k window to process long contracts/reports without context truncation.
  - **Fine-grained Attribute Discrimination**: Distinguishing "Large" vs. "Small" (attribute-level nuances) through contrastive fine-tuning.

## 2. ModernBERT Advantages (Benchmark Analysis)

ModernBERT is chosen for its superior performance across key metrics. For detailed definitions of these benchmarks, see the [Benchmarks Guide](../../../../05_Evaluation/01_Benchmarks_Taxonomy.md).

| Category          | Metric (ModernBERT Large) | Performance Status           |
| :---------------- | :------------------------ | :--------------------------- |
| **Search (BEIR)** | **44.0** (DPR)            | Top-tier Zero-shot Retrieval |
| **Search (MLDR)** | **80.4** (ColBERT)        | Unrivaled Long-context IR    |
| **NLU (GLUE)**    | **90.4**                  | Competitive with DeBERTaV3   |
| **Code (CSN)**    | **59.5**                  | High Structural Logic        |

- **Flexibility (8k Context)**: Native support for 8k context (vs. BERT's 512). It supports **Linear Scaling** via RoPE, allowing extension to 32k+ with minimal compute cost.
- **Efficiency (GeGLU)**: Uses Gated Linear Units for smoother gradient flow in structural text (Code/Math), reducing Loss Spikes during initial training.
- **Speed (Hardware-Aware)**: Built for **FlashAttention-2** and **Unpadding** out of the box, optimized for modern GPU kernels.

## 3. High-Performance Architecture Practices

### 3.1 Sequence Packing & Unpadding (Varlen Interface)

Traditional BERT wastes 50%+ compute on Padding. ModernBERT supports **Unpadding**:

- **Strategy**: Concatenate all tokens in a batch into a 1D sequence using the `varlen` interface.
- **Result**: **40% lower VRAM usage** and **3x speedup** when processing long sequences (8k).

### 3.2 Matryoshka Representation Learning (MRL)

For massive-scale retrieval systems (Search/Ads):

- **Strategy**: Add MRL to the loss function to force the model to capture core semantics in the first 64/128/256 dimensions.
- **Result**: Enables **Fast 128d Retrieval** followed by **Precise 1024d Reranking** within a single model architecture.

### 3.3 Hard-Negative DPO for Encoders

To solve "Semantic Drift" in cases like "Large vs. Small":

- **Strategy**: Construct triplets $(Anchor, Positive, Hard\_Negative)$. Use an **InfoNCE Loss** variant to penalize hard negatives (antonyms with high cosine similarity).
- **Goal**: "Tear apart" logical opposites in the manifold space.

## 3. Best Practices

- **Pooling Strategy**: In 8k long-context scenarios, discard `[CLS]` and use **Attention-Weighted Pooling** to prevent information decay at the sequence end.
- **Unpadding**: Mandatory for speed. Use the `unpad_sequences` utility from the `modernbert` library.
- **Precision Management**: Native to **BF16**. If using FP16 (e.g., on specific local hardware), add a **StableEmbedding** layer to prevent RoPE-induced numerical overflow.
- **Learning Rate**: 2e-5 to 5e-5 with a linear warmup of 10%.

---

## 4. Key References

1. **Devlin et al. (2019)**: *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.
2. **Warner et al. (2024)**: *Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Fine-tuning and Inference* (ModernBERT).
