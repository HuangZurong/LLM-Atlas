# PEFT (Parameter-Efficient Fine-Tuning)

*Prerequisite: [01_Introduction.md](01_Introduction.md).*

---

Traditional full-parameter fine-tuning is computationally prohibitive for models over 7B. PEFT allows us to achieve similar performance by updating only a tiny fraction of parameters.

### A Brief History of PEFT

- **2019**: Adapter Tuning (Houlsby et al.) — the start of parameter-efficient learning.
- **2021**: The "Big Year" — Prefix Tuning, Prompt Tuning, P-Tuning V1/V2, and **LoRA**.
- **2023**: QLoRA — consumer-grade training for massive models.
- **2024**: DoRA — bridging the gap between LoRA and full fine-tuning.

---

## 1. LoRA (Low-Rank Adaptation)

**Core idea**: Freeze pretrained weights $W_0$ and inject a trainable low-rank decomposition $\Delta W = BA$.

$$h = W_0 x + \Delta W x = W_0 x + BAx$$

Where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d, k)$.

**Initialization**: $A$ is initialized with Kaiming uniform; $B$ is initialized to zero — so $\Delta W = 0$ at the start of training (the model begins from the pretrained weights).

**Scaling factor**: The update is scaled by $\alpha / r$, where $\alpha$ is a hyperparameter:
$$h = W_0 x + \frac{\alpha}{r} BAx$$

### 1.1 Rank Selection Guidelines

| Rank $r$ | Trainable Params (per layer) | Use Case |
|:--|:--|:--|
| 4–8 | Very small | Simple tasks, limited data |
| 16–32 | Moderate | General SFT, instruction tuning (most common) |
| 64–128 | Larger | Complex domain adaptation, multilingual |

**Rule of thumb**: Start with $r=16$, $\alpha=32$. Increase rank if validation loss plateaus.

### 1.2 Which Layers to Target

| Target | Effect |
|:--|:--|
| Q, V only | Original LoRA paper default; good baseline |
| Q, K, V, O | Better for instruction tuning; standard in practice |
| Q, K, V, O + FFN (gate, up, down) | Best quality; approaches full fine-tuning |

### 1.3 Inference: Zero Overhead

After training, merge $\Delta W$ back into $W_0$:
$$W_{merged} = W_0 + \frac{\alpha}{r} BA$$

The merged model has **identical architecture and latency** to the original — no adapter overhead at inference.

---

## 2. DoRA (Weight-Decomposed Low-Rank Adaptation) [2024]

- **Insight**: Decomposes pretrained weight $W$ into **magnitude** $m = \|W\|_c$ and **direction** $V = W / \|W\|_c$ (column-wise norms). Applies LoRA only to the directional component $V$.
- **Why it works**: Full fine-tuning naturally adjusts magnitude and direction independently. Standard LoRA entangles them. DoRA decouples them, closing the accuracy gap.
- **Usage**: `use_dora=True` in HuggingFace `LoraConfig`. Same VRAM as LoRA.

---

## 3. QLoRA

- **Method**: Quantize the base model to 4-bit (NF4 data type) and add LoRA adapters in BF16. Gradients flow through the quantized weights via a dequantization step.
- **Key innovations** (Dettmers et al., 2023):
  - **NF4 (Normal Float 4)**: Information-theoretically optimal 4-bit quantization for normally distributed weights.
  - **Double quantization**: Quantize the quantization constants themselves, saving ~0.37 bits/param.
  - **Paged optimizers**: Use CPU memory as overflow for optimizer states via unified memory.
- **Impact**: 65B model trainable on a single 48GB GPU. VRAM reduction ~70% vs BF16 LoRA.

---

## 4. Adapter Tuning

- **Concept**: Insert small bottleneck modules (down-project → nonlinearity → up-project) after attention/FFN sublayers.
- **Architecture**: $\text{Adapter}(x) = x + f(x W_{down}) W_{up}$, where $W_{down} \in \mathbb{R}^{d \times m}$, $W_{up} \in \mathbb{R}^{m \times d}$, $m \ll d$.
- **Trainable params per adapter**: $2md + d + m$ (including biases).
- **Tradeoff vs LoRA**: Adapters add inference latency (cannot be merged); LoRA has zero inference overhead. LoRA is now the dominant choice.

---

## 5. Prefix & Prompt Tuning (Soft Prompts)

- **Prefix Tuning** (Li & Liang, 2021): Prepend learnable continuous vectors to the key/value pairs at **every attention layer**. The model and its weights are completely frozen.
- **Prompt Tuning** (Lester et al., 2021): A simplified version — learnable vectors prepended only at the **input embedding layer**. Effectiveness matches full fine-tuning at model scales >10B.
- **P-Tuning V2** (Liu et al., 2022): Inserts continuous prompts across every layer (like Prefix Tuning) but optimized for encoder models (GLM, BERT).

---

## 6. Method Comparison

| Method | Trainable Params | Inference Overhead | VRAM | Best For |
|:--|:--|:--|:--|:--|
| **LoRA** | ~0.1–1% | **None** (mergeable) | Low | General SFT, most use cases |
| **DoRA** | ~0.1–1% | **None** (mergeable) | Low | When LoRA quality is insufficient |
| **QLoRA** | ~0.1–1% | None (after merge) | **Very low** | Limited GPU memory |
| **Adapter** | ~1–5% | Small (extra layers) | Low | Legacy; largely superseded by LoRA |
| **Prefix/Prompt** | <0.1% | Small (extra tokens) | Very low | Multi-task serving (swap prefixes) |

---

## 7. Key References

1. **Hu et al. (2021)**: *LoRA: Low-Rank Adaptation of Large Language Models*.
2. **Dettmers et al. (2023)**: *QLoRA: Efficient Finetuning of Quantized Language Models*.
3. **Liu et al. (2024)**: *DoRA: Weight-Decomposed Low-Rank Adaptation*.
4. **Houlsby et al. (2019)**: *Parameter-Efficient Transfer Learning for NLP* (Adapter Tuning).
5. **Li & Liang (2021)**: *Prefix-Tuning: Optimizing Continuous Prompts for Generation*.
6. **Lester et al. (2021)**: *The Power of Scale for Parameter-Efficient Prompt Tuning*.
