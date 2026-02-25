# Training Frontiers

*Prerequisite: [../03_Pre_Training/](../03_Pre_Training/). Continuously updated. Last update: 2025-06*

---

## 1. New Scaling Law Findings

### [Chinchilla Revisited] Scaling Data-Constrained Language Models (Muennighoff et al., 2023)
- **Institution**: Hugging Face / BigScience
- **Link**: arXiv:2305.16264
- **One-line Summary**: When high-quality data is insufficient, repeating data still helps but with diminishing returns — nearly zero gain after 4 epochs.
- **Core Innovation**: Proposes a data-constrained scaling law quantifying the impact of data repetition on loss.
- **Implications**: Complements Chinchilla's limitations in data-scarce scenarios.

### [DeepSeek-V3 Scaling] DeepSeek-V3 Technical Report (2024)
- **Institution**: DeepSeek
- **Link**: arXiv:2412.19437
- **One-line Summary**: 671B MoE model trained with only 2.788M H800 GPU hours (~$5.5M), demonstrating extreme efficiency of MoE + MLA + FP8.
- **Core Innovation**: Auxiliary-loss-free load balancing (bias term replaces auxiliary loss); Multi-Token Prediction as training objective.
- **Key Results**: 14.8T tokens trained; performance matches GPT-4o / Claude 3.5 Sonnet.

---

## 2. Data Engineering

### [FineWeb] FineWeb: Decanting the Web for the Finest Text Data (Penedo et al., 2024)
- **Institution**: Hugging Face
- **Link**: arXiv:2406.17557
- **One-line Summary**: 15T-token high-quality English web dataset with fully documented processing pipeline from CommonCrawl to final data.
- **Core Innovation**: Systematic ablation of each filtering step (URL filtering, language detection, quality classification, deduplication) and its independent contribution.
- **Key Results**: Models trained on FineWeb consistently outperform those trained on C4, RefinedWeb, and Dolma.
- **Implications**: Validates the data processing pipeline described in 02_Dataset.

### [DCLM] DataComp-LM: In Search of the Next Generation of Training Sets (Li et al., 2024)
- **Institution**: DataComp Consortium (multi-institution)
- **Link**: arXiv:2406.11794
- **One-line Summary**: Open data filtering competition framework proving that data selection strategy matters more than data scale.
- **Core Innovation**: Fixes model architecture and training config, varies only data filtering methods for fair comparison.
- **Key Results**: Best filtering strategy improves 7B model MMLU from 25.8% to 64.0%.

---

## 3. Training Efficiency

### [DeepGEMM] DeepGEMM: Clean and Efficient FP8 GEMM Kernels (DeepSeek, 2025)
- **Institution**: DeepSeek
- **Link**: github.com/deepseek-ai/DeepGEMM
- **One-line Summary**: JIT-compiled FP8 matrix multiplication kernels matching or exceeding expert hand-written CUDA kernels without pre-compilation.
- **Core Innovation**: Leverages CUDA JIT compilation to dynamically generate optimal kernels based on runtime matrix shapes.

### [DeepEP] DeepEP: Efficient Expert-Parallel Communication (DeepSeek, 2025)
- **Institution**: DeepSeek
- **Link**: github.com/deepseek-ai/DeepEP
- **One-line Summary**: High-efficiency All-to-All communication library for MoE models, supporting both low-latency (inference) and high-throughput (training) modes.
- **Core Innovation**: NVLink + RDMA hybrid communication with GPU SMs directly managing network transfers.

### [FP8 Training] FP8-LM: Training FP8 Large Language Models (Peng et al., 2023)
- **Institution**: Microsoft
- **Link**: arXiv:2310.18313
- **One-line Summary**: Extends FP8 from inference to training — forward, backward, and optimizer states all in FP8.
- **Core Innovation**: Mixed Precision 3.0 framework — forward FP8, backward FP8, master weights FP32, optimizer states FP8.
- **Key Results**: GPT-175B training memory reduced 39%, speed improved 75%, accuracy loss <0.5%.

---

## 4. Frontier Training Paradigms (2025)

### [Llama 4 Training] Llama 4 Training at Scale (Meta, 2025)
- **Institution**: Meta
- **Link**: arXiv:2601.11659
- **One-line Summary**: Trains Llama 4 Behemoth (2T total params) using Early Fusion multimodal pre-training — text and images jointly from the start, not post-hoc adapter.
- **Core Innovation**: Native multimodal pre-training eliminates the modality gap; iRoPE enables 10M context without RoPE extrapolation artifacts.
- **Implications**: Shifts the paradigm from "LLM + vision adapter" to "multimodal from day one."

### [DeepSeek-R1-0528] DeepSeek-R1-0528 Update (DeepSeek, 2025)
- **Institution**: DeepSeek
- **Link**: huggingface.co/deepseek-ai/DeepSeek-R1-0528
- **One-line Summary**: Major update to R1 with enhanced post-training algorithms; average reasoning tokens doubled from 12K to 23K per question.
- **Core Innovation**: Extended fine-tuning with increased compute budget + algorithmic optimization for deeper chain-of-thought reasoning.
- **Key Results**: Matches OpenAI o3 and Gemini 2.5 Pro on AIME 2024 and MATH-500; 685B parameters (up from 671B).

### [Qwen3 Training] Qwen3 Large-Scale MoE Training (Alibaba, 2025)
- **Institution**: Alibaba Qwen
- **One-line Summary**: Trains 235B-A22B MoE across 36T tokens with 4-stage curriculum: general → STEM → code → reasoning alignment.
- **Core Innovation**: Hybrid thinking mode trained via two-phase approach — first standard SFT, then RL-based thinking mode with reward from verifiable tasks.
- **Key Results**: Full model family from 0.6B to 235B, all open-weight; 30B-A3B achieves strong results at extreme efficiency.
