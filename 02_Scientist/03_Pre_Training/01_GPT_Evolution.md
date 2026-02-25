# Foundation Model Evolution: From BERT to Llama 3

*Prerequisite: [../01_Architecture/01_Transformer.md](../01_Architecture/01_Transformer.md).*
*See Also: [09_Research_Trends.md](09_Research_Trends.md) (2025 industry trends), [../../03_Engineering/06_Deployment/01_Theory/01_Optimization.md](../../03_Engineering/06_Deployment/01_Theory/01_Optimization.md) (inference optimization).*

---

## 1. The Great Convergence: Why Decoder-Only?

The early Transformer era (2018-2020) was a battle between three paradigms:
1. **Encoder-only** (BERT, RoBERTa): Best for NLU (classification, NER).
2. **Encoder-Decoder** (T5, BART): Best for transduction (translation, summarization).
3. **Decoder-only** (GPT series): Best for generation.

### 1.1 The Efficiency/Capability Tradeoff
| Paradigm | Training Efficiency | Zero-shot Generalization | Why it won/lost |
|:--|:--|:--|:--|
| **Encoder-only** | High | Low | Bidirectional attention leaks information, preventing efficient causal scaling. |
| **Encoder-Decoder** | Medium | Medium | Redundant parameters; the cross-attention bottleneck hinders massive scaling. |
| **Decoder-only** | **Highest** | **Highest** | Causal masking allows every token to be a training signal. Emergent ICL (In-Context Learning) is strongest here. |

### 1.2 Mathematical Insight: The Unified Objective
Modern research shows that Decoder-only models are essentially **Prefix-LM** capable. By training on a causal objective, the model learns the joint distribution $P(x_1, \dots, x_n)$, which implicitly contains the conditional distributions needed for all other tasks.

---

## 2. GPT Family: The Scaling Pioneer (OpenAI)

### 2.1 GPT-1 to GPT-3: The Proof of Scaling
- **GPT-1 (117M)**: Proved that generative pre-training provides a "universal representation" that beats supervised baselines on 9/12 tasks.
- **GPT-2 (1.5B)**: Demonstrated **Zero-shot Task Transfer**. The model could perform translation or summarization without being told how, simply by following the prompt's statistical flow.
- **GPT-3 (175B)**: The "In-Context Learning" breakthrough. Proved that scaling parameters alone can unlock few-shot reasoning without weight updates.

### 2.2 GPT-4: The Multi-Modal Reasoning Leap
While GPT-3 was a dense transformer, **GPT-4** moved to a **Sparse Mixture of Experts (MoE)**:
- **Total Parameters**: ~1.8 Trillion.
- **Active Parameters**: ~280 Billion per token.
- **Lesson**: Scaling "Capacity" (total params) is more important than scaling "Compute" (active params) for world knowledge.

---

## 3. Llama: The Open-Weight Standard (Meta AI)

### 3.1 Llama 1 & 2: Democratizing Foundation Models
Meta moved the industry toward a standard recipe:
- **Pre-norm**: Using RMSNorm for stability.
- **SwiGLU**: Better non-linearity than GeLU.
- **RoPE**: Better position handling than absolute embeddings.

### 3.2 Llama 3: The Data-Centric Breakthrough (2024)
Llama 3 (8B, 70B, 405B) changed the Scaling Law narrative:
- **Finding**: Even an 8B model can continue to improve up to **15 Trillion tokens**.
- **Implication**: Most previous models were **vastly under-trained**. The "Chinchilla Limit" is a compute-optimal point, but not a capability-optimal point.

---

## 4. BERT and the NLU Legacy (Google)

### 4.1 Masked Language Modeling (MLM)
BERT's contribution was the **Bidirectional Context**:
$$L_{MLM} = -\sum \log P(x_{masked} | x_{unmasked})$$
This remains the gold standard for dense retrieval (embedding models like **BGE** or **ModernBERT**).

---

## 5. Comparative Evolution Summary

| Generation | Representative Model | Key Architectural Shift | Primary Capability |
|:--|:--|:--|:--|
| **1st Gen** | BERT / GPT-1 | Basic Transformer | Contextual Embeddings |
| **2nd Gen** | GPT-2 / T5 | Scaling to 1B+ | Zero-shot Transfer |
| **3rd Gen** | GPT-3 / PaLM | Scaling to 100B+ | In-Context Learning |
| **4th Gen** | Llama 2 / Mistral | Efficiency (GQA/SWA) | Open-weight Parity |
| **5th Gen** | GPT-4 / Llama 3 | MoE & 15T+ Tokens | Human-level Reasoning |

---

## 6. Key References

1. **Radford et al. (2018)**: *Improving Language Understanding by Generative Pre-Training* (GPT-1).
2. **Devlin et al. (2019)**: *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.
3. **Brown et al. (2020)**: *Language Models are Few-Shot Learners* (GPT-3).
4. **Touvron et al. (2023)**: *Llama: Open and Efficient Foundation Language Models*.
5. **Meta AI (2024)**: *The Llama 3 Herd of Models*.

---

*Document upgraded to Scientist/Researcher standard in 2025.*
