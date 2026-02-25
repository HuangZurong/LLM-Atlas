# Multimodal Frontiers

*Prerequisite: [../06_Multimodal/](../06_Multimodal/). Continuously updated. Last update: 2025-06*

---

## 1. Unified Multimodal Architectures

### [Gemini 1.5] Gemini 1.5: Unlocking Multimodal Understanding Across Millions of Tokens (Reid et al., 2024)
- **Institution**: Google DeepMind
- **Link**: arXiv:2403.05530
- **One-line Summary**: Native multimodal MoE model supporting 1M+ token context, processing text, images, audio, and video simultaneously.
- **Core Innovation**: No separate vision/audio encoders — all modalities processed in a unified Transformer; ultra-long context via efficient attention.
- **Key Results**: >99% NIAH recall at 1M context; experimental validation at 10M tokens.
- **Implications**: Represents the "unified native multimodal" route, contrasting with LLaVA-style "modular assembly."

### [GPT-4o] GPT-4o System Card (OpenAI, 2024)
- **Institution**: OpenAI
- **Link**: openai.com/index/gpt-4o-system-card
- **One-line Summary**: Native multimodal model with end-to-end text/image/audio input and output, enabling real-time voice conversation.
- **Core Innovation**: Audio processed directly inside the model without ASR/TTS intermediate steps, preserving tone, emotion, and prosody.
- **Relation to Existing Work**: Same "native multimodal" camp as Gemini 1.5, but architecture details undisclosed.

---

## 2. Vision-Language Models

### [InternVL 2.5] Expanding Performance Boundaries of Open-Source Multimodal Models (Chen et al., 2024)
- **Institution**: Shanghai AI Lab
- **Link**: arXiv:2412.05271
- **One-line Summary**: Latest open-source VLM series matching GPT-4o on multiple benchmarks.
- **Core Innovation**: Dynamic resolution (AnyRes) + Pixel Shuffle compression + large-scale multimodal data.
- **Key Results**: MMMU 72.0%, MathVista 72.3%, OCRBench 882.

### [Qwen2-VL] Qwen2-VL: Enhancing Vision-Language Model's Perception (Wang et al., 2024)
- **Institution**: Alibaba Qwen
- **Link**: arXiv:2409.12191
- **One-line Summary**: Introduces 3D RoPE (spatial width × spatial height × temporal depth) for native arbitrary-resolution image and video support.
- **Core Innovation**: Naive Dynamic Resolution (no predefined tile grid) + 3D RoPE unifying image/video positional encoding.
- **Key Results**: 2B/7B/72B scales; 72B version surpasses GPT-4o on Video-MME.
- **Implications**: Documented in 06_Multimodal/03_Video_Understanding.md (3D RoPE section).

---

## 3. Video Understanding

### [Video-LLM Survey] Video Understanding with Large Language Models: A Survey (Tang et al., 2024)
- **Institution**: Multi-institution survey
- **Link**: arXiv:2312.17432
- **One-line Summary**: Comprehensive survey of video-language model architectures, training, and evaluation methods.
- **Core Innovation**: Systematically categorizes three video understanding paradigms: Frame-level, Temporal Aggregation, and Native Long-Context.
- **Implications**: Validates the architecture taxonomy in 06_Multimodal/03_Video_Understanding.md.

---

## 4. Speech & Audio

### [Whisper v3] Whisper Large-v3 (OpenAI, 2023)
- **Institution**: OpenAI
- **Link**: huggingface.co/openai/whisper-large-v3
- **One-line Summary**: Latest Whisper version with expanded training data and improved multilingual performance.
- **Key Results**: 128-channel Mel spectrogram (v2 used 80-channel); significantly reduced multilingual WER on FLEURS.

### [VALL-E 2] VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot TTS (Chen et al., 2024)
- **Institution**: Microsoft
- **Link**: arXiv:2406.05370
- **One-line Summary**: First zero-shot TTS system achieving human parity on LibriSpeech.
- **Core Innovation**: Repetition Aware Sampling + Grouped Code Modeling, solving VALL-E 1's repetition and stability issues.
- **Key Results**: First to reach human parity on speaker similarity and robustness.

---

## 5. Image Generation

### [Flux] FLUX.1: Next-Generation Image Generation (Black Forest Labs, 2024)
- **Institution**: Black Forest Labs (original Stable Diffusion team)
- **Link**: blackforestlabs.ai
- **One-line Summary**: Rectified Flow + DiT architecture image generation model surpassing SDXL and Midjourney v5 in text rendering and composition.
- **Core Innovation**: Flow Matching replaces DDPM; Transformer replaces UNet; native text rendering support.

---

## 6. 2025 Multimodal Updates

### [Gemini 2.5 Pro] Gemini 2.5 Pro (Google DeepMind, 2025)
- **Institution**: Google DeepMind
- **Link**: blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025
- **One-line Summary**: Thinking model with native multimodal reasoning across text, images, audio, and video; 1M-token context (expanding to 2M).
- **Core Innovation**: Integrates "thinking" (internal CoT) with native multimodal processing — can reason about video frames, audio segments, and code simultaneously.
- **Key Results**: #1 on LMArena at launch; top scores on MMMU, MathVista, and video understanding benchmarks.
- **Implications**: Combines the reasoning model trend (o1/R1) with native multimodality (Gemini 1.5).

### [Llama 4 Multimodal] Llama 4 Early Fusion (Meta, 2025)
- **Institution**: Meta
- **Link**: arXiv:2601.11659
- **One-line Summary**: First open-weight model family with native multimodal pre-training (Early Fusion) — text and images processed jointly from the start.
- **Core Innovation**: No separate vision encoder or adapter; all modalities share the same Transformer backbone from pre-training, eliminating the modality gap.
- **Key Results**: Scout and Maverick accept interleaved text-image inputs natively; competitive with GPT-4o on multimodal benchmarks.
- **Relation to Existing Work**: Contrasts with LLaVA/InternVL "modular" approach; aligns with Gemini's "native" philosophy but open-weight.

### [GPT-4o Native Audio] GPT-4o Audio Capabilities (OpenAI, 2025)
- **Institution**: OpenAI
- **Link**: openai.com/index/gpt-4o-system-card
- **One-line Summary**: Full native audio input/output in production — real-time voice conversation with emotion, tone, and multilingual code-switching.
- **Core Innovation**: End-to-end audio processing without ASR/TTS pipeline; model directly generates audio tokens preserving prosody and speaker characteristics.
- **Implications**: Sets the standard for voice-native AI assistants; competitors (Gemini 2.5, Qwen2-Audio) following similar native audio direction.
