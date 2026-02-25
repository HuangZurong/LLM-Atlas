# Vision-Language Models: From CLIP to GPT-4V

*Prerequisite: [../01_Architecture/01_Transformer.md](../01_Architecture/01_Transformer.md). Covers LLaVA, Qwen-VL, and GPT-4V architectures.*
*See Also: [../07_Paper_Tracking/05_Multimodal_Frontiers.md](../07_Paper_Tracking/05_Multimodal_Frontiers.md) (latest multimodal papers), [../../04_Solutions/09_Vertical_Scenario_Templates.md](../../04_Solutions/09_Vertical_Scenario_Templates.md) (multimodal in industry verticals).*

---

## 1. The VLM Architecture Paradigm

All modern Vision-Language Models (VLMs) share a three-component design:

```
Image → [Vision Encoder] → [Connector/Projector] → [LLM Backbone] → Text
```

### 1.1 Vision Encoders
| Encoder | Resolution | Patch Size | Output | Used By |
| :--- | :--- | :--- | :--- | :--- |
| **CLIP ViT-L/14** | 336×336 | 14×14 | 576 tokens | LLaVA 1.5 |
| **EVA-CLIP** | 448×448 | 14×14 | 1024 tokens | InternVL |
| **SigLIP-So400m** | 224 / 448 | 14×14 | 256 / 1024 tokens | PaliGemma |
| **Native ViT** | Dynamic | 14×14 | Variable | GPT-4V |

> **Crucial Detail**: Most VLMs (e.g., LLaVA) use features from the **penultimate layer** (倒数第二层) of the vision encoder, as the final layer is too specialized for contrastive matching and lacks spatial detail needed for reasoning.

### 1.2 Connector Architectures
The connector bridges the vision and language representation spaces:
- **Linear Projection** (LLaVA v1): Single linear layer $W \in \mathbb{R}^{d_{vis} \times d_{llm}}$. Simple but effective.
- **MLP Projector** (LLaVA v1.5): Two-layer MLP with GELU activation. Standard choice.
- **Cross-Attention Resampler** (Flamingo/Qwen-VL): Uses learnable queries to compress visual tokens to a fixed count (e.g., 256 → 64). Reduces compute cost.
- **C-Abstractor** (Honeybee): Convolutional abstractor preserving spatial locality.

---

## 2. Training Paradigm: Two-Stage

### 2.1 Stage 1: Pre-training (Alignment)
- **Goal**: Align vision encoder output space with LLM input space.
- **Data**: Large-scale image-caption pairs (e.g., LAION-5B, CC-12M).
- **What's trained**: Only the connector. Vision encoder and LLM are frozen.

### 2.2 Stage 2: Instruction Tuning
- **Goal**: Teach the model to follow visual instructions (VQA, OCR, reasoning).
- **Data**: High-quality visual instruction data (LLaVA-Instruct-150K, ShareGPT4V).
- **What's trained**: Connector + LLM (often via LoRA). Vision encoder typically stays frozen.

---

## 3. Key Architectural Innovations

### 3.1 Dynamic Resolution (AnyRes)
Fixed-resolution encoders waste compute on simple images and lose detail on complex ones. **AnyRes** (LLaVA-NeXT, InternVL 1.5):
1. Divide the input image into tiles matching the encoder's native resolution.
2. Encode each tile independently.
3. Concatenate all tile tokens + a downscaled global view.
- **Impact**: Enables OCR and fine-grained document understanding.

### 3.2 Visual Token Compression
High-resolution images produce thousands of visual tokens, overwhelming the LLM context. Solutions:
- **Pixel Shuffle** (InternVL 2): Spatially merge adjacent tokens, reducing count by 4x.
- **Perceiver Resampler** (Flamingo): Fixed-length output regardless of input resolution.

---

## 4. Industrial Case Studies

### 4.1 LLaVA Series (Open Source)
- **LLaVA 1.0** (2023): Proved that a simple linear projection + visual instruction tuning is surprisingly effective.
- **LLaVA 1.5**: MLP projector + higher resolution (336px) → significant gains.
- **LLaVA-NeXT**: AnyRes + more data → competitive with proprietary models.

### 4.2 GPT-4V / GPT-4o (OpenAI)
Architecture not disclosed. Key capabilities:
- Native multi-image understanding.
- Interleaved image-text reasoning.
- OCR and spatial reasoning at production quality.

### 4.3 Qwen-VL (Alibaba)
- Cross-attention resampler with 256 learnable queries.
- Supports bounding box grounding (outputting coordinates).

---

## 5. Key References

1.  **Liu et al. (2023)**: *Visual Instruction Tuning* (LLaVA).
2.  **Alayrac et al. (2022)**: *Flamingo: a Visual Language Model for Few-Shot Learning*.
3.  **Bai et al. (2023)**: *Qwen-VL: A Versatile Vision-Language Model*.
4.  **Chen et al. (2023)**: *InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks* (v1 report; 2024 for v1.5/v2).
