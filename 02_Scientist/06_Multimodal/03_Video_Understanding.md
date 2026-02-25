# Video Understanding: Temporal Reasoning at Scale

*Prerequisite: [01_Vision_Language_Models.md](01_Vision_Language_Models.md). Covers Video-LLaVA, Gemini 1.5, and temporal reasoning.*

---

## 1. The Core Challenge

Video = Images + Time. The key difficulty is **temporal reasoning** — understanding how events unfold across frames — while managing the massive token count (a 1-minute video at 1 FPS with 256 tokens/frame = 15,360 tokens).

---

## 2. Frame Sampling Strategies

| Strategy | Description | Trade-off |
| :--- | :--- | :--- |
| **Uniform sampling** | Select $N$ frames evenly spaced | Simple; misses fast events |
| **Keyframe extraction** | Scene-change detection to pick informative frames | Better coverage; more complex |
| **Dense sampling** | High FPS, subsample tokens per frame | Best temporal fidelity; very expensive |

---

## 3. Architectural Approaches

### 3.1 Frame-Level Encoding (Most Common)
Sample $N$ frames → encode each independently with a ViT → concatenate all visual tokens → feed to LLM.
- **Used by**: Video-LLaVA, LLaVA-NeXT-Video.
- **Limitation**: No explicit temporal modeling between frames.

### 3.2 Temporal Aggregation
Add temporal modules between the vision encoder and LLM:
- **Temporal Attention**: Cross-attention across frames at the same spatial position.
- **Video Q-Former** (Video-BLIP2): Learnable queries attend to all frames simultaneously, compressing temporal information.
- **SlowFast**: Dual-stream processing at different temporal resolutions.

### 3.3 Temporal Encoding Innovations
- **3D RoPE (Rotary Positional Embeddings)**: Models like **Qwen2-VL** extend RoPE to three dimensions (spatial width, spatial height, and temporal depth). This allows the model to handle variable resolutions and frame rates natively.
- **Native Long-Context**: Models like **Gemini 1.5 Pro** handle video by treating frames as part of a massive context window (1M+ tokens). No special temporal module needed — the Transformer's attention handles temporal relationships implicitly.

---

## 4. Key Tasks

| Task | Description | Benchmark |
| :--- | :--- | :--- |
| **Video QA** | Answer questions about video content | ActivityNet-QA, MSVD-QA |
| **Video Captioning** | Generate descriptions of video events | MSR-VTT |
| **Temporal Grounding** | Locate when an event occurs in a video | Charades-STA |
| **Long Video Understanding** | Reason over hour-long videos | EgoSchema, Video-MME |

---

## 5. Key References

1.  **Lin et al. (2023)**: *Video-LLaVA: Learning United Visual Representation by Alignment Before Projection*.
2.  **Reid et al. (2024)**: *Gemini 1.5: Unlocking Multimodal Understanding Across Millions of Tokens of Context*.
3.  **Li et al. (2023)**: *VideoChat: Chat-Centric Video Understanding*.
