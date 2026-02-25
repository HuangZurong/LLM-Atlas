# Track 2: The Scientist (Model Development)

This track covers the core research and engineering required to build and adapt Large Language Models — from architecture design through training, alignment, evaluation, and multimodal extension.

---

## Module Map

### [01. Architecture](./01_Architecture/)

| File | Topic |
|:--|:--|
| [01_Transformer.md](./01_Architecture/01_Transformer.md) | Transformer fundamentals: MHA, KV Cache, Decoder-only design |
| [02_Attention.md](./01_Architecture/02_Attention.md) | Attention in depth: MHA, GQA, MQA, **MLA** (DeepSeek-V2) |
| [03_Efficient_Attention.md](./01_Architecture/03_Efficient_Attention.md) | FlashAttention, sparse attention variants |
| [04_Tokenizer.md](./01_Architecture/04_Tokenizer.md) | BPE, WordPiece, Unigram algorithms and implementation |
| [05_Embedding.md](./01_Architecture/05_Embedding.md) | Vocabulary expansion, initialization, training flows |
| [06_Position_Encoding.md](./01_Architecture/06_Position_Encoding.md) | Absolute, Relative, RoPE, ALiBi |
| [07_Architecture_Paradigms.md](./01_Architecture/07_Architecture_Paradigms.md) | Encoder-only, Decoder-only, Encoder-Decoder |
| [08_Dense_Arch.md](./01_Architecture/08_Dense_Arch.md) | Dense model design (LLaMA, GPT, etc.) |
| [09_MoE_Arch.md](./01_Architecture/09_MoE_Arch.md) | Mixture of Experts: routing, load balancing, DeepSeek MoE |
| [10_Decoding.md](./01_Architecture/10_Decoding.md) | Decoding strategies: greedy, beam search, sampling |
| [11_Interpretability.md](./01_Architecture/11_Interpretability.md) | Mechanistic interpretability: probing, SAE, circuits, superposition |
| [12_Long_Context.md](./01_Architecture/12_Long_Context.md) | Long context: RoPE scaling (PI/NTK/YaRN/ABF), Ring Attention, KV Cache management |

### [02. Dataset](./02_Dataset/)

| File | Topic |
|:--|:--|
| [01_Pre_Training_Data_at_Scale.md](./02_Dataset/01_Pre_Training_Data_at_Scale.md) | 15T+ token pipeline: cleaning, deduplication (MinHash/Suffix Array), quality filtering |
| [02_Instruction_Data_Engineering.md](./02_Dataset/02_Instruction_Data_Engineering.md) | Self-Instruct, Evol-Instruct, IFD scoring |
| [03_Preference_Data_Construction.md](./02_Dataset/03_Preference_Data_Construction.md) | Human annotation, RLAIF, Constitutional AI data |
| [04_Synthetic_Data_Engineering.md](./02_Dataset/04_Synthetic_Data_Engineering.md) | Synthetic data generation and quality control |
| [05_Data_Safety_and_PII_Management.md](./02_Dataset/05_Data_Safety_and_PII_Management.md) | PII detection, layered defense, compliance |

### [03. Pre-Training](./03_Pre_Training/)

| File | Topic |
|:--|:--|
| [01_GPT_Evolution.md](./03_Pre_Training/01_GPT_Evolution.md) | GPT-1 → GPT-4: the generative revolution |
| [02_Scaling_Laws.md](./03_Pre_Training/02_Scaling_Laws.md) | Kaplan/Chinchilla laws, compute-optimal training |
| [03_Transformer_Variants.md](./03_Pre_Training/03_Transformer_Variants.md) | Architectural innovations: SwiGLU, RMSNorm, MTP |
| [04_Attention_Optimizations.md](./03_Pre_Training/04_Attention_Optimizations.md) | FlashAttention, PagedAttention, kernel-level optimization |
| [05_Data_Pipelines.md](./03_Pre_Training/05_Data_Pipelines.md) | Data loading, tokenization, curriculum scheduling |
| [06_Optimization_Techniques.md](./03_Pre_Training/06_Optimization_Techniques.md) | AdamW, FP8 mixed precision, gradient accumulation |
| [07_Distributed_Training.md](./03_Pre_Training/07_Distributed_Training.md) | 3D Parallelism (DP/TP/PP), DeepSpeed, FSDP |
| [08_Training_Stability.md](./03_Pre_Training/08_Training_Stability.md) | Loss spike detection, recovery, numerical stability |
| [09_Research_Trends.md](./03_Pre_Training/09_Research_Trends.md) | SSM/Mamba, hybrid architectures, efficiency frontiers |
| [10_Pre_Training_Evaluation.md](./03_Pre_Training/10_Pre_Training_Evaluation.md) | Pre-training evaluation metrics and monitoring |
| [11_Continual_Pre_Training.md](./03_Pre_Training/11_Continual_Pre_Training.md) | Continual pre-training: domain/language adaptation, catastrophic forgetting |

### [04. Post-Training](./04_Post_Training/)

**Fine-Tuning** ([01_FT/](./04_Post_Training/01_FT/))

| File | Topic |
|:--|:--|
| [01_Introduction.md](./04_Post_Training/01_FT/01_Theory/01_Introduction.md) | Fine-tuning fundamentals |
| [02_PEFT_Strategies.md](./04_Post_Training/01_FT/01_Theory/02_PEFT_Strategies.md) | LoRA, DoRA, QLoRA, Adapter, Prefix Tuning |
| [01_ModernBERT.md](./04_Post_Training/01_FT/02_Encoder_NLU/01_ModernBERT_FT/01_ModernBERT.md) | Encoder fine-tuning: ModernBERT, classification |
| [01_LLM_SFT_Guide.md](./04_Post_Training/01_FT/03_Decoder_NLG/01_LLM_SFT_Guide.md) | Decoder SFT: chat templates, prompt masking, TRL |
| [01_Domain_Adaptation.md](./04_Post_Training/01_FT/04_Techniques/01_Domain_Adaptation.md) | Domain adaptation: Chinese-LLaMA case study |

**Alignment** ([02_Alignment/](./04_Post_Training/02_Alignment/))

| File | Topic |
|:--|:--|
| [01_Overview.md](./04_Post_Training/02_Alignment/01_Overview.md) | 4-stage paradigm: SFT → Preference → Reasoning alignment |
| [01_Overview.md](./04_Post_Training/02_Alignment/02_Preference_Alignment/01_Overview.md) | RLHF pipeline, reward modeling |
| [02_PPO.md](./04_Post_Training/02_Alignment/02_Preference_Alignment/02_PPO.md) | Proximal Policy Optimization |
| [03_DPO.md](./04_Post_Training/02_Alignment/02_Preference_Alignment/03_DPO.md) | Direct Preference Optimization |
| [04_KTO.md](./04_Post_Training/02_Alignment/02_Preference_Alignment/04_KTO.md) | Kahneman-Tversky Optimization |
| [05_RLAIF.md](./04_Post_Training/02_Alignment/02_Preference_Alignment/05_RLAIF.md) | RL from AI Feedback |
| [06_Constitutional_AI.md](./04_Post_Training/02_Alignment/02_Preference_Alignment/06_Constitutional_AI.md) | Rule-based self-critique |
| [07_Safety_Fine_Tuning.md](./04_Post_Training/02_Alignment/02_Preference_Alignment/07_Safety_Fine_Tuning.md) | Safety-specific alignment |
| [01_RLVR.md](./04_Post_Training/02_Alignment/03_Reasoning_Alignment/01_RLVR.md) | RL with Verifiable Rewards |
| [02_GRPO.md](./04_Post_Training/02_Alignment/03_Reasoning_Alignment/02_GRPO.md) | Group Relative Policy Optimization (DeepSeek-R1) |
| [01_Rejection_Sampling.md](./04_Post_Training/02_Alignment/04_Advanced_Topics/01_Rejection_Sampling.md) | Best-of-N sampling |
| [02_Iterative_Training.md](./04_Post_Training/02_Alignment/04_Advanced_Topics/02_Iterative_Training.md) | Online and iterative alignment |
| [03_Inference_Time_Compute.md](./04_Post_Training/02_Alignment/04_Advanced_Topics/03_Inference_Time_Compute.md) | Scaling compute at inference (o1/R1 paradigm) |
| [04_Model_Merging.md](./04_Post_Training/02_Alignment/04_Advanced_Topics/04_Model_Merging.md) | Merging aligned models |

**Distillation** ([03_Distillation/](./04_Post_Training/03_Distillation/))

| File | Topic |
|:--|:--|
| [01_Overview.md](./04_Post_Training/03_Distillation/01_Overview.md) | KD taxonomy, R1-Distill case study, logit vs data distillation |

### [05. Evaluation](./05_Evaluation/)

| File | Topic |
|:--|:--|
| [01_Benchmarks_Taxonomy.md](./05_Evaluation/01_Benchmarks_Taxonomy.md) | MMLU, GPQA, GSM8K, HumanEval, SWE-bench, long context, multilingual |
| [02_Evaluation_Methodology.md](./05_Evaluation/02_Evaluation_Methodology.md) | pass@k, Elo vs BT-MLE, statistical rigor, benchmark saturation |
| [03_LLM_as_Judge.md](./05_Evaluation/03_LLM_as_Judge.md) | MT-Bench, AlpacaEval 2.0 (LC), Arena-Hard, known biases |
| [04_Safety_Evaluation.md](./05_Evaluation/04_Safety_Evaluation.md) | TruthfulQA, BBQ, red-teaming (GCG, AutoDAN, PAIR) |
| [05_Contamination_Detection.md](./05_Evaluation/05_Contamination_Detection.md) | N-gram overlap, MIN-K% PROB, perturbation-based detection |

### [06. Multimodal](./06_Multimodal/)

| File | Topic |
|:--|:--|
| [01_Vision_Language_Models.md](./06_Multimodal/01_Vision_Language_Models.md) | VLM architecture: encoders, connectors, AnyRes, LLaVA/GPT-4V/Qwen-VL |
| [02_Audio_Speech_Models.md](./06_Multimodal/02_Audio_Speech_Models.md) | Whisper, VALL-E, semantic vs acoustic tokens, GPT-4o |
| [03_Video_Understanding.md](./06_Multimodal/03_Video_Understanding.md) | Frame sampling, temporal aggregation, 3D RoPE |
| [04_Multimodal_Evaluation.md](./06_Multimodal/04_Multimodal_Evaluation.md) | MMMU, MMBench, Video-MME, hallucination (POPE/CHAIR) |

### [07. Paper Tracking](./07_Paper_Tracking/)

| File | Topic |
|:--|:--|
| [01_Tracking_Methodology.md](./07_Paper_Tracking/01_Tracking_Methodology.md) | Sources, filtering criteria, three-pass reading framework |
| [02_Architecture_Frontiers.md](./07_Paper_Tracking/02_Architecture_Frontiers.md) | Mamba/Hybrid, MLA, NSA, Ring Attention, YaRN |
| [03_Training_Frontiers.md](./07_Paper_Tracking/03_Training_Frontiers.md) | Data-constrained scaling, FineWeb, DCLM, FP8, DeepGEMM |
| [04_Alignment_Frontiers.md](./07_Paper_Tracking/04_Alignment_Frontiers.md) | R1, o1, SimPO, ORPO, RepE, Circuit Breakers |
| [05_Multimodal_Frontiers.md](./07_Paper_Tracking/05_Multimodal_Frontiers.md) | Gemini 1.5, GPT-4o, InternVL, Qwen2-VL, VALL-E 2 |

---

*Deployment and operations modules are in [Track 3: Engineering](../03_Engineering/). Classic paper deep-dives will be maintained in a separate top-level directory.*
