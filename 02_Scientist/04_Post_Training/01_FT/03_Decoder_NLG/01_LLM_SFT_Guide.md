# LLM Supervised Fine-Tuning (Decoder-only NLG)

*Prerequisite: [../01_Theory/01_Introduction.md](../01_Theory/01_Introduction.md).*

> Reference: [Understanding and Using Supervised Fine-Tuning (SFT) for Language Models](https://cameronrwolfe.substack.com/p/understanding-and-using-supervised) — Cameron R. Wolfe, 2023

---

## 1. Where SFT Fits in the LLM Training Pipeline

Modern LLMs follow a three-step alignment framework, first systematized by InstructGPT:

```
Pre-training  ──→  SFT  ──→  RLHF / DPO
 (Raw text)     (Curated)   (Preference)
```

| Stage | Data | Objective |
|:--|:--|:--|
| Pre-training | Massive unlabeled corpus | Learn language distribution (Next Token Prediction) |
| **SFT** | **High-quality instruction-response pairs** | **Align with human instruction preferences** |
| RLHF / DPO | Human preference annotations (chosen vs rejected) | Further refine alignment |

**Key distinction from Pre-training**: The training objective is identical (both use Next Token Prediction / Cross Entropy). The difference lies in the data — SFT uses carefully curated high-quality examples rather than raw web text.

## 2. Core Mechanics

### 2.1 Training Objective

SFT computes loss **only on the model's response tokens**; the prompt/instruction portion is masked out:

```
[System] You are a helpful assistant.  ← masked, no loss
[User] What is LoRA?                   ← masked, no loss
[Assistant] LoRA is a parameter-       ← loss computed here
efficient fine-tuning method that...   ← loss computed here
```

### 2.2 Full Fine-Tuning vs. PEFT

| Method | Parameters Updated | VRAM | Use Case |
|:--|:--|:--|:--|
| Full FT | All weights | Very high | Large data, domain saturation |
| LoRA / DoRA | Low-rank adapters | Moderate | Industry standard for 7B+ models |
| QLoRA | 4-bit quantization + LoRA | Lowest | Consumer-grade GPU training |

### 2.3 Data Formatting

- **Chat Templates**: Jinja templates separating `system`, `user`, and `assistant` roles.
- **Prompt Masking**: Ensures the model only learns from response tokens, not instruction tokens.
- **Formats**: Alpaca format (`instruction` / `input` / `output`) or ShareGPT multi-turn conversation format.

## 3. Pros and Cons of SFT

### Pros

1. **Simple to implement**: Same Next Token Prediction objective as pre-training; low code complexity.
2. **Computationally efficient**: ~100X cheaper than pre-training.
3. **Highly effective**: Dramatically improves instruction following, output consistency, and coherence.
4. **Preserves generality**: The model does not over-specialize to a single task.

### Cons

1. **Data quality is critical**: Results depend heavily on curated dataset quality; manual inspection does not scale.
2. **RLHF still adds value**: Research shows further refinement via RLHF yields substantial improvements beyond SFT alone.
3. **Diversity > Quantity**: Data quality and diversity outweigh raw dataset size.

### Current Consensus

> Moderate-sized, high-quality SFT datasets + human preference data for RLHF refinement = optimal path.

## 4. Key Research Findings

### 4.1 LIMA: Quality > Quantity

**[LIMA (2023)](https://arxiv.org/abs/2305.11206)** demonstrated that only **1,000** carefully curated examples can achieve results competitive with top models. Takeaways:

- Manual curation ensures diversity and coverage.
- Data quality matters far more than quantity.
- A small number of high-quality examples beats a large volume of low-quality data.

### 4.2 Imitation Learning

Fine-tuning open-source models by collecting dialogue data from proprietary models (ChatGPT / GPT-4):

| Model | Method | Result |
|:--|:--|:--|
| Alpaca | 52K examples via Self-Instruct + GPT-3.5 | Solid performance at low cost |
| Vicuna | ShareGPT conversation data | 90%+ of ChatGPT quality |
| Koala | Mixed dialogue data | Open-source community benchmark |
| Orca | Large-scale GPT-4 reasoning traces | Pushed the ceiling of imitation learning |

### 4.3 The InstructGPT Framework

InstructGPT was the first to publicly disclose the full SFT + RLHF training pipeline — a key reference for understanding modern LLM alignment:

1. Collect high-quality human-written demonstration data.
2. SFT on the base model.
3. Collect human preference ranking data.
4. Train a Reward Model.
5. Optimize the policy model with PPO.

### 4.4 LLaMA-2 in Practice

LLaMA-2 used **27,540** SFT samples + RLHF, optimizing along both safety and helpfulness dimensions simultaneously.

## 5. Hyperparameter Reference

| Parameter | Recommended | Notes |
|:--|:--|:--|
| Batch Size (global) | 128 – 512 | Large batches help stabilize training |
| Learning Rate | 1e-5 ~ 2e-5 | Typically lower than pre-training |
| Scheduler | Cosine with warmup | Standard choice |
| Weight Decay | 0.1 | Prevents overfitting |
| Epochs | 2 – 5 | SFT datasets are small; watch for overfitting |

## 6. Tooling

### TRL (Transformer Reinforcement Learning)

HuggingFace's TRL library provides `SFTTrainer` for straightforward SFT:

```python
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./sft_output",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-5,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

Key features:
- **Prompt Masking**: Compute loss only on response tokens.
- **Packing**: Concatenate multiple short samples into one sequence to maximize GPU utilization.
- **PEFT integration**: Pass a LoRA config directly for parameter-efficient fine-tuning.

## 7. Public SFT Datasets

| Dataset | Size | Characteristics |
|:--|:--|:--|
| Dolly 15K | 15K | Hand-written by Databricks employees |
| Alpaca | 52K | Self-Instruct + GPT-3.5 generated |
| ShareGPT (Vicuna) | ~70K | Real user conversations with ChatGPT |
| UltraChat | 1.5M | Large-scale multi-turn dialogue |
| OpenAssistant (OASST) | 161K | Multilingual, human-annotated |
| Baize | 111K | ChatGPT self-chat data |

---

## 8. Key References

1. **Ouyang et al. (2022)**: *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT).
2. **Taori et al. (2023)**: *Stanford Alpaca: An Instruction-following LLaMA Model*.
3. **von Werra et al. (2023)**: *TRL: Transformer Reinforcement Learning* (HuggingFace TRL library).
