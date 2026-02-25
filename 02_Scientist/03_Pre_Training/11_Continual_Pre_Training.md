# Continual Pre-Training: Adapting Foundation Models Without Starting Over

*Prerequisite: [02_Scaling_Laws.md](02_Scaling_Laws.md), [06_Optimization_Techniques.md](06_Optimization_Techniques.md).*

---

## 1. Motivation

Training a frontier LLM from scratch costs millions of dollars and months of compute. Continual pre-training (CPT) allows:

| Goal | Example |
|:--|:--|
| **Domain adaptation** | Adapt a general model to medicine, law, or finance |
| **Knowledge update** | Incorporate new world knowledge without full retraining |
| **Language adaptation** | Extend an English-centric model to Chinese, Japanese, etc. |
| **Capability extension** | Add code, math, or long-context abilities to an existing model |

CPT sits between full pre-training and fine-tuning — it uses the pre-training objective (next-token prediction) but starts from an existing checkpoint rather than random initialization.

---

## 2. Catastrophic Forgetting

The central challenge: when a model trains on new data distribution $D_{new}$, it degrades on the original distribution $D_{orig}$.

### 2.1 Why It Happens

- Gradient updates optimized for $D_{new}$ overwrite weights that encoded $D_{orig}$ knowledge.
- The more $D_{new}$ diverges from $D_{orig}$, the worse the forgetting.
- Smaller models forget faster than larger ones (larger models have more capacity to absorb new knowledge without overwriting).

### 2.2 Measuring Forgetting

Track performance on a held-out benchmark suite from $D_{orig}$ throughout CPT:

$$\text{Forgetting}_t = \frac{\text{Perf}_{orig}(t=0) - \text{Perf}_{orig}(t)}{\text{Perf}_{orig}(t=0)}$$

Common benchmarks: MMLU (general knowledge), HellaSwag (commonsense), HumanEval (code).

---

## 3. Mitigation Strategies

### 3.1 Data Replay (Mixing)

The most effective and widely used approach. Mix new domain data with a fraction of the original pre-training data:

$$D_{CPT} = \alpha \cdot D_{new} + (1 - \alpha) \cdot D_{replay}$$

**Typical ratios**:
- Domain adaptation: $\alpha = 0.5 \text{–} 0.8$ (50–80% new data)
- Knowledge update: $\alpha = 0.3 \text{–} 0.5$ (more replay to preserve breadth)

**Key findings**:
- Even 10–20% replay dramatically reduces forgetting (Scialom et al., 2022).
- The replay data should be **representative** of the original distribution, not just random samples.
- Quality matters more than quantity — use the same filtered/curated data from original pre-training if available.

### 3.2 Learning Rate Schedule

Starting CPT with the original pre-training's final learning rate (or slightly above) is critical:

| Strategy | Description |
|:--|:--|
| **Re-warmup + cosine decay** | Brief warmup to $\eta_{max}^{CPT}$ (typically 10–50% of original $\eta_{max}$), then cosine decay |
| **Constant low LR** | Use a small constant LR (~1e-5 to 5e-5) throughout CPT |

**Anti-pattern**: Using the original high learning rate destroys pre-trained representations immediately.

### 3.3 Regularization-Based Methods

| Method | Mechanism |
|:--|:--|
| **EWC** (Elastic Weight Consolidation) | Penalize changes to weights important for previous tasks (measured by Fisher information) |
| **L2 regularization** | $\mathcal{L}_{reg} = \lambda \|\theta - \theta_{orig}\|_2^2$ — keep weights close to original |
| **Knowledge distillation** | Use the original model as teacher; add KL divergence loss on logits |

In practice, **data replay + careful LR scheduling** outperforms regularization methods for LLM-scale CPT. EWC and similar methods add significant memory overhead and are rarely used at scale.

### 3.4 Architecture-Based Approaches

| Method | Mechanism | Trade-off |
|:--|:--|:--|
| **LoRA-based CPT** | Train low-rank adapters instead of full weights | Less forgetting but limited capacity for large distribution shifts |
| **Mixture of Experts expansion** | Add new expert modules for new domain | No forgetting (original experts frozen) but increases model size |
| **Progressive layer freezing** | Freeze early layers, train later layers | Preserves low-level features; limits adaptation depth |

---

## 4. Domain-Adaptive Pre-Training (DAPT)

### 4.1 The Standard Recipe

Gururangan et al. (2020) — *Don't Stop Pretraining*:

1. Start from a general-purpose checkpoint.
2. Continue pre-training on domain-specific corpus with next-token prediction.
3. Then fine-tune on downstream tasks.

**Result**: DAPT consistently improves downstream task performance, even when the domain corpus is relatively small (tens of millions of tokens).

### 4.2 Notable Examples

| Model | Base | Domain | Corpus Size | Key Result |
|:--|:--|:--|:--|:--|
| **BioMedLM** | GPT-2 architecture | Biomedical | 34.6B tokens (PubMed) | Competitive with much larger general models on medical QA |
| **CodeLlama** | Llama 2 | Code | 500B code tokens | Outperforms Llama 2 on code tasks while retaining general ability |
| **Llemma** | Code Llama | Mathematics | 55B tokens (Proof-Pile-2) | State-of-the-art open math model at release |
| **SaulLM** | Mistral 7B | Legal | 30B legal tokens | Significant gains on legal benchmarks |

### 4.3 Lessons Learned

- **More data > more epochs**: Repeating domain data beyond 2–3 epochs yields diminishing returns (consistent with Muennighoff et al., 2023).
- **Data quality is paramount**: Filtered, deduplicated domain data outperforms raw crawls by a large margin.
- **Tokenizer mismatch**: If the domain has specialized vocabulary (e.g., chemical formulas, legal citations), consider extending the tokenizer with domain-specific tokens before CPT.

---

## 5. Language-Adaptive Pre-Training

Extending a primarily English model to other languages.

### 5.1 Approach

1. **Extend tokenizer**: Add tokens for the target language to improve fertility (tokens per word). A pure English tokenizer may fragment Chinese/Japanese text into individual bytes.
2. **Initialize new embeddings**: New token embeddings can be initialized by averaging semantically similar existing embeddings, or trained from scratch.
3. **Continue pre-training**: Mix target-language data with English replay data.

### 5.2 Key Considerations

- **Embedding layer**: Resizing the embedding matrix requires careful initialization — random init for new tokens causes a temporary loss spike.
- **Cross-lingual transfer**: Models retain surprising cross-lingual ability after CPT, especially for typologically similar languages.
- **Replay ratio**: Higher English replay (40–60%) is needed to preserve English capability compared to domain CPT.

---

## 6. Long-Context Continual Pre-Training

A specific and increasingly common form of CPT (see also [12_Long_Context.md](../01_Architecture/12_Long_Context.md)):

1. Adjust RoPE base frequency (ABF) or apply YaRN scaling.
2. Continue pre-training on long documents (32K–128K tokens) with reduced learning rate.
3. Typically 1–5% of original pre-training compute.

**Examples**: Llama 3.1 (8K → 128K), Qwen2 (4K → 128K), Yi (4K → 200K).

---

## 7. Practical Guidelines

| Decision | Recommendation |
|:--|:--|
| **When to use CPT vs. fine-tuning** | CPT when the domain shift is large and you need broad domain knowledge; fine-tuning when you have specific task data |
| **Compute budget** | Typically 1–10% of original pre-training cost |
| **Data volume** | Minimum ~1B tokens for meaningful domain adaptation; 10B+ for strong results |
| **Evaluation** | Always track both domain performance AND general benchmarks throughout training |
| **Checkpointing** | Save frequently — the optimal checkpoint often isn't the final one |

---

## 8. Key References

1. **Gururangan et al. (2020)**: *Don't Stop Pretraining: Adapt Pretrained Language Models to Domains and Tasks*.
2. **Scialom et al. (2022)**: *Fine-Tuned Language Models are Continual Learners*.
3. **Rozière et al. (2023)**: *Code Llama: Open Foundation Models for Code* — Meta.
4. **Azerbayev et al. (2023)**: *Llemma: An Open Language Model for Mathematics*.
5. **Muennighoff et al. (2023)**: *Scaling Data-Constrained Language Models* — data repetition limits.
6. **Ibrahim et al. (2024)**: *Simple and Scalable Strategies to Continually Pre-train Large Language Models* — Meta.
7. **Gupta et al. (2023)**: *Continual Pre-Training of Large Language Models: How to Re-warm Your Model?*
