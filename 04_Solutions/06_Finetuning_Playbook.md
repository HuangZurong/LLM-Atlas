# Fine-tuning Playbook: Planning and Executing Domain Adaptation

*Prerequisite: [../02_Scientist/04_Post_Training/01_FT/](../02_Scientist/04_Post_Training/01_FT/).*

---

This document is a project-level guide for fine-tuning LLMs on domain data. For the theory of fine-tuning methods (LoRA, RLHF, DPO), see 02_Scientist/04_Post_Training. Here we focus on: how to plan, execute, and iterate a fine-tuning project from start to finish.

## 1. Should You Fine-tune at All?

Fine-tuning is expensive and irreversible (in the sense that you commit resources). Before starting, verify that fine-tuning is actually the right solution.

### 1.1 Fine-tuning Solves These Problems

- The model doesn't understand domain terminology (e.g., confuses "punch list" with "to-do list")
- The model's output style is wrong (too casual, too verbose, wrong format)
- The model can't follow domain-specific reasoning patterns
- The model needs to consistently produce structured output (JSON, tables, reports)
- Inference latency from RAG is unacceptable

### 1.2 Fine-tuning Does NOT Solve These Problems

- The model lacks specific factual knowledge → Use RAG instead
- The model hallucinates → Fine-tuning can reduce but not eliminate this; RAG with grounding is more effective
- You want the model to access real-time data → Use tools/APIs
- You have fewer than 500 quality examples → Use few-shot prompting first

### 1.3 The Litmus Test

Ask yourself: "Can I solve this with a better prompt?" If yes, don't fine-tune. Fine-tune only when prompt engineering has hit its ceiling and you can clearly articulate what behavior change you need.

## 2. Project Planning

### 2.1 Define Success Criteria Before Training

| Criterion | Bad Example | Good Example |
| :--- | :--- | :--- |
| **Accuracy** | "Model should be accurate" | "Model answers domain Q&A with >85% accuracy on our 200-question eval set" |
| **Style** | "Model should sound professional" | "Model outputs follow our report template with all 5 required sections" |
| **Safety** | "Model shouldn't say bad things" | "Model refuses to provide advice outside its domain in >95% of adversarial tests" |
| **Latency** | "Model should be fast" | "P95 latency < 2 seconds for single-turn Q&A on our target hardware" |

### 2.2 Resource Planning

| Component | Options | Estimated Cost |
| :--- | :--- | :--- |
| **Base model** | 7B (QLoRA on single 24GB GPU) / 14B (QLoRA on 2×24GB) / 72B (multi-GPU cluster) | GPU rental: $1-50/hour |
| **Training data** | Expert authoring ($50-200/hour) + LLM-assisted generation ($0.01-0.10/example) | 5K examples ≈ $500-5000 |
| **Compute** | Cloud GPU (A100/H100) or local workstation | 7B QLoRA: ~$10-50 per training run |
| **Evaluation** | Expert reviewers for output quality assessment | $500-2000 per evaluation round |
| **Iteration** | Plan for 3-5 training cycles minimum | Total: 3-5× single run cost |

### 2.3 Timeline Template

| Phase | Activities | Typical Duration |
| :--- | :--- | :--- |
| **Phase 1: Baseline** | Evaluate base model on domain tasks, establish metrics | Week 1 |
| **Phase 2: Data v1** | Collect and prepare initial training data | Week 2-3 |
| **Phase 3: Train v1** | First fine-tuning run, evaluate results | Week 3-4 |
| **Phase 4: Error analysis** | Categorize failures, identify data gaps | Week 4 |
| **Phase 5: Data v2** | Create targeted data for failure categories | Week 5 |
| **Phase 6: Train v2** | Second training run with improved data | Week 5-6 |
| **Phase 7: Iterate** | Repeat phases 4-6 until success criteria met | Week 6-8 |

## 3. Base Model Selection for Fine-tuning

### 3.1 Current Landscape (2024-2025)

| Model | Parameters | License | Strengths | Fine-tuning Ecosystem |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen 2.5** | 0.5B-72B | Apache 2.0 | Strong multilingual, good Chinese | Excellent (HF, vLLM, LLaMA-Factory) |
| **Llama 3.1/3.2** | 1B-405B | Llama License | Strong English, large community | Excellent (widest tool support) |
| **Mistral/Mixtral** | 7B-8x22B | Apache 2.0 | Efficient, good reasoning | Good (HF, vLLM) |
| **DeepSeek V2/V3** | 16B-236B (MoE) | MIT | Strong coding and reasoning | Growing |
| **GLM-4** | 9B | Apache 2.0 | Good Chinese, tool use | Moderate |

### 3.2 Selection Criteria

1. **Language**: If your domain is primarily Chinese, prioritize Qwen or GLM. For English, Llama or Mistral.
2. **Size vs hardware**: Match model size to your available GPU. QLoRA on 7B needs ~10GB VRAM; 14B needs ~20GB; 72B needs ~40GB+ (quantized).
3. **Base capability**: Test the base model (without fine-tuning) on your domain tasks. A model that already performs at 60% is easier to fine-tune to 85% than one starting at 30%.
4. **License**: Verify the license permits your use case (commercial, academic, government).

## 4. Training Data Preparation

### 4.1 Data Format

Standard chat format (compatible with most fine-tuning frameworks):

```json
{
  "messages": [
    {"role": "system", "content": "You are a construction operations expert specializing in infrastructure project management."},
    {"role": "user", "content": "What factors should be considered when planning the transition from construction to operations phase?"},
    {"role": "assistant", "content": "The construction-to-operations transition requires careful planning across several dimensions:\n\n1. **Commissioning readiness**: All systems must be tested and verified...\n2. **Staff training**: Operations personnel need to be trained on...\n3. **Documentation handover**: As-built drawings, O&M manuals...\n4. **Warranty management**: Track warranty periods for all major equipment...\n5. **Performance benchmarks**: Establish baseline metrics for..."}
  ]
}
```

### 4.2 Data Composition

A well-balanced fine-tuning dataset should include:

| Category | Proportion | Purpose |
| :--- | :--- | :--- |
| **Domain Q&A** | 40-50% | Core domain knowledge and reasoning |
| **Task-specific examples** | 20-30% | Report generation, classification, extraction |
| **General capability preservation** | 10-20% | Prevent catastrophic forgetting of general skills |
| **Safety and refusal** | 5-10% | Teach the model to decline out-of-scope requests |
| **Edge cases** | 5-10% | Ambiguous queries, multi-step reasoning, conflicting information |

### 4.3 Data Quality Checklist

Before training, verify:
- [ ] No data leakage between training and evaluation sets
- [ ] Answers are factually correct (expert-verified sample)
- [ ] Consistent formatting across examples
- [ ] Balanced coverage of sub-domains
- [ ] System prompts are consistent
- [ ] No PII in training data
- [ ] Sufficient diversity in question types and difficulty levels

## 5. Training Configuration

### 5.1 QLoRA Configuration (Recommended Default)

```python
# LoRA hyperparameters
lora_config = {
    "r": 64,              # Rank. 16-128. Higher = more capacity, more memory
    "lora_alpha": 128,     # Scaling factor. Typically 2× r
    "lora_dropout": 0.05,  # Regularization
    "target_modules": [    # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"        # FFN
    ],
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# Training hyperparameters
training_config = {
    "num_train_epochs": 3,          # 2-5 epochs for SFT
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,  # Effective batch size = 4 × 4 = 16
    "learning_rate": 2e-4,           # 1e-4 to 5e-4 for QLoRA
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "max_grad_norm": 1.0,
    "bf16": True,                    # Use bf16 if hardware supports
    "max_seq_length": 2048,          # Match your data distribution
}

# Quantization
quantization_config = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_use_double_quant": True
}
```

### 5.2 Key Hyperparameter Decisions

| Parameter | Conservative | Aggressive | Guidance |
| :--- | :--- | :--- | :--- |
| **Learning rate** | 1e-4 | 5e-4 | Start conservative, increase if loss plateaus |
| **Epochs** | 2 | 5 | Monitor eval loss; stop if it increases (overfitting) |
| **LoRA rank** | 16 | 128 | Higher rank for more complex domain adaptation |
| **Batch size** | 8 | 32 | Larger = more stable gradients, needs more memory |
| **Max seq length** | 1024 | 4096 | Match your data. Longer = more memory, slower |

### 5.3 Training Frameworks

| Framework | Strengths | Best For |
| :--- | :--- | :--- |
| **LLaMA-Factory** | GUI + CLI, many models, easy config | Beginners, rapid experimentation |
| **Axolotl** | Flexible YAML config, good defaults | Intermediate users |
| **TRL (HuggingFace)** | Official, well-documented, SFT + DPO + RLHF | Full training pipeline |
| **Unsloth** | 2x faster training, lower memory | Resource-constrained environments |

## 6. Evaluation and Iteration

### 6.1 Evaluation Protocol

After each training run:

1. **Automated metrics**: Run the model on your evaluation set. Compute accuracy, BLEU/ROUGE (if applicable), format compliance rate.
2. **LLM-as-judge**: Use a stronger model (GPT-4, Claude) to rate outputs on relevance, accuracy, completeness, and style. Score 1-5 on each dimension.
3. **Human evaluation**: Domain experts review a sample (50-100 outputs). This is the ground truth.
4. **A/B comparison**: Compare new model vs previous version on the same questions. Which is better, and why?

### 6.2 Error Taxonomy

Categorize every failure into actionable categories:

| Error Type | Example | Fix |
| :--- | :--- | :--- |
| **Knowledge gap** | Model doesn't know about a specific regulation | Add training examples covering this topic |
| **Reasoning error** | Model applies wrong logic to a scenario | Add chain-of-thought examples for similar scenarios |
| **Format violation** | Output doesn't follow required structure | Add more examples with correct formatting |
| **Hallucination** | Model invents plausible but false details | Add examples where model should say "I don't know" |
| **Catastrophic forgetting** | Model lost general capabilities after fine-tuning | Add general-purpose examples to training mix |
| **Overfitting** | Model memorizes training examples verbatim | Reduce epochs, increase data diversity |

### 6.3 The Iteration Loop

```
Train → Evaluate → Error Analysis → Targeted Data Creation → Retrain
  ↑                                                              |
  └──────────────────────────────────────────────────────────────┘
```

Each iteration should:
1. Fix the top 3-5 error categories from the previous round
2. Add 500-2000 new targeted examples
3. Retrain (often from the base model, not from the previous checkpoint, to avoid compounding errors)
4. Re-evaluate on the full eval set plus new test cases for the targeted categories

## 7. Deployment Considerations

### 7.1 Model Merging

After QLoRA training, you have a base model + adapter weights. For deployment:

- **Keep separate**: Load base + adapter at inference time. Flexible (can swap adapters) but slightly slower.
- **Merge**: Combine adapter into base model weights. Simpler deployment, no adapter overhead. Use this for production.

### 7.2 Quantization for Deployment

| Method | Size Reduction | Quality Loss | Speed |
| :--- | :--- | :--- | :--- |
| **FP16** | 2× vs FP32 | Negligible | Baseline |
| **GPTQ (4-bit)** | 4× vs FP16 | Small (1-2% on benchmarks) | Fast (GPU) |
| **AWQ (4-bit)** | 4× vs FP16 | Small | Fast (GPU) |
| **GGUF (4-bit)** | 4× vs FP16 | Small | CPU-friendly |

### 7.3 Serving Infrastructure

| Tool | Strengths | Best For |
| :--- | :--- | :--- |
| **vLLM** | PagedAttention, high throughput, continuous batching | Production GPU serving |
| **Ollama** | Simple setup, local deployment | Development, single-user |
| **TGI (HuggingFace)** | Production-ready, good monitoring | HuggingFace ecosystem |
| **llama.cpp** | CPU inference, edge deployment | Resource-constrained environments |

## 8. Multi-Stage Fine-tuning Strategy

For complex domain applications, a single round of SFT is often insufficient. Consider a staged approach:

### Stage 1: Continued Pre-training (CPT)

- **Goal**: Teach the model domain vocabulary and concepts
- **Data**: Large domain corpus (10M-100M+ tokens), unstructured text
- **Method**: Standard causal language modeling (next token prediction)
- **Outcome**: Model understands domain terminology but doesn't follow instructions yet

### Stage 2: Supervised Fine-tuning (SFT)

- **Goal**: Teach the model to follow domain-specific instructions
- **Data**: Instruction pairs (5K-20K examples)
- **Method**: Chat-format training with system/user/assistant roles
- **Outcome**: Model can answer domain questions in the desired format

### Stage 3: Preference Alignment (RL & Alignment)

- **Goal**: Align the model with human experts on subjective quality, safety, and reasoning depth.
- **Methods**:
  - **DPO (Direct Preference Optimization)**: **Primary recommendation**. No reward model needed; directly optimizes the model based on (Chosen, Rejected) pairs. Stable and efficient.
  - **PPO (Proximal Policy Optimization)**: Requires a separate Reward Model (RM). Use only if you need complex, multi-objective reinforcement learning (e.g., maximizing truthfulness while maintaining helpfulness).
  - **KTO (Kahneman-Tversky Optimization)**: Only needs binary feedback (Good/Bad) instead of pairs. Best when preference data is noisy or unpaired.
  - **ORPO (Odds Ratio Preference Optimization)**: Combines SFT and Alignment into one stage.

**Alignment Data Strategy:**
- **Contrastive Pairs**: "Given Query X, Response A is better than B because it follows technical standard GB-5001."
- **Reasoning Chains (CoT)**: Prefer responses that show their work.
- **Safety Boundary**: Rejecting out-of-scope or dangerous prompts (e.g., "How to bypass a safety lock?").

### Stage 4: Knowledge Injection vs. Reasoning Adaptation

| Strategy | When to use | Data focus |
| :--- | :--- | :--- |
| **Domain Adaptation (CPT)** | Low overlap with general world knowledge. | Large-scale raw domain corpus. |
| **Logic & Tool Use (SFT)** | Needs to use calculators, DBs, or specific APIs. | Multi-turn tool-call trajectories. |
| **Precision Alignment (DPO)** | Model is too chatty or lacks professional tone. | Expert-ranked comparison pairs. |

---

## Key References

1. **Hu et al. (2021)**: *LoRA: Low-Rank Adaptation of Large Language Models*.
2. **Dettmers et al. (2023)**: *QLoRA: Efficient Finetuning of Quantized Language Models*.
