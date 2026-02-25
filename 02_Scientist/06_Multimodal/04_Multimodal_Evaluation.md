# Multimodal Evaluation: Benchmarking Beyond Text

*Prerequisite: [01_Vision_Language_Models.md](01_Vision_Language_Models.md), [../05_Evaluation/01_Benchmarks_Taxonomy.md](../05_Evaluation/01_Benchmarks_Taxonomy.md).*

---

## 1. Vision-Language Benchmarks

| Benchmark | What it Tests | Format | Notes |
| :--- | :--- | :--- | :--- |
| **MMMU** | College-level multimodal reasoning (charts, diagrams, art) | MCQ | 11.5K questions; **MMMU-Pro** (2024) uses harder Grad-level tasks. |
| **MMBench** | Comprehensive VLM ability (perception + reasoning) | MCQ | CircularEval to reduce position bias |
| **LMSYS Chatbot Arena (Vision)** | Human preference ranking for VLMs | Voting | The current "Gold Standard" for production quality |
| **MathVista** | Mathematical reasoning with visual inputs | Free-form | Charts, geometry, function plots |
| **OCRBench** | Text recognition in natural images | Exact match | Tests OCR + layout understanding |
| **RealWorldQA** | Real-world spatial reasoning from photos | MCQ | Autonomous driving scenarios |

---

## 2. Video Benchmarks

| Benchmark | What it Tests | Format | Notes |
| :--- | :--- | :--- | :--- |
| **Video-MME** | Comprehensive video understanding | MCQ | Short/Medium/Long video splits |
| **EgoSchema** | Long-form egocentric video QA | MCQ | 5K 3-minute clips (~250 total hours) |
| **MVBench** | 20 temporal reasoning tasks | MCQ | Action sequence, scene transition |

---

## 3. Audio Benchmarks

| Benchmark | What it Tests | Format | Notes |
| :--- | :--- | :--- | :--- |
| **LibriSpeech** | English ASR | WER | Clean/Other splits |
| **FLEURS** | Multilingual ASR (102 languages) | WER | The "multilingual Whisper" test |
| **AudioCaps** | Audio captioning | CIDEr/METEOR | Describe what you hear |

---

## 4. Hallucination in VLMs

A critical failure mode: the model "sees" things that aren't in the image.
- **POPE (Polling-based Object Probing)**: Ask "Is there a [object] in the image?" for objects that are/aren't present. Measures hallucination rate.
- **CHAIR**: Measures the fraction of generated caption objects that don't exist in the image.
- **Mitigation**: Higher-resolution encoders and better visual grounding reduce hallucination.

---

## 5. Key References

1.  **Yue et al. (2023)**: *MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark*.
2.  **Liu et al. (2023)**: *MMBench: Is Your Multi-modal Model an All-around Player?*
3.  **Li et al. (2023)**: *Evaluating Object Hallucination in Large Vision-Language Models* (POPE).
