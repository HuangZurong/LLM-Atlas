# Audio & Speech Models: From Whisper to GPT-4o

*Prerequisite: [01_Vision_Language_Models.md](01_Vision_Language_Models.md) (shared VLM concepts). Covers Whisper, AudioPaLM, and GPT-4o.*

---

## 1. Audio Representation

### 1.1 Semantic vs. Acoustic Tokens
Modern audio models (e.g., SpeechGPT, Gemini) distinguish between two types of tokens:
- **Semantic Tokens**: Capture the "content" or "meaning" of speech (e.g., from Whisper or HuBERT). Invariant to speaker or emotion.
- **Acoustic Tokens**: Capture the "prosody," "timbre," and "emotion" (e.g., from EnCodec or SoundStream). Essential for TTS and natural voice cloning.

### 1.2 Key Encoders
- **Whisper Encoder**: CNN + Transformer on 80-channel log-Mel spectrograms. 30-second chunks. **Whisper v3** expanded training data and improved large-v3 performance.
- **HuBERT**: Self-supervised encoder producing discrete units via k-means clustering.
- **wav2vec 2.0**: Contrastive learning on raw waveforms → contextualized representations.

---

## 2. Speech Recognition (ASR)

### 2.1 Whisper Architecture
```
Audio (30s) → Mel Spectrogram → Encoder (Transformer) → Cross-Attention → Decoder (Transformer) → Text
```
- **Trained on**: 680K hours of weakly-supervised web audio.
- **Key strength**: Extreme robustness to accents, noise, and languages (99 languages).
- **Limitation**: 30-second chunking loses long-range context.

### 2.2 Domain Adaptation
Standard Whisper fails on specialized jargon (medical, legal). Solutions:
- **LoRA fine-tuning** on domain-specific transcripts (most efficient).
- **Prompt conditioning**: Providing domain context in the decoder prefix.

---

## 3. Speech Synthesis (TTS)

### 3.1 Neural Codec Language Models
The 2024-2025 frontier: treat speech as a "language" of discrete codec tokens.
- **VALL-E (Microsoft)**: Encodes speech into EnCodec tokens, then uses an autoregressive Transformer to generate speech from text + 3-second voice prompt.
- **Zero-shot voice cloning**: Generate any voice from a short sample.

### 3.2 End-to-End Models
- **GPT-4o**: Natively processes audio input and generates audio output without a separate ASR/TTS pipeline. This enables real-time conversation with natural prosody and emotion.

---

## 4. Audio Understanding (Beyond ASR)

Modern audio-language models go beyond transcription:
- **Audio captioning**: "Describe what you hear" → "A dog barking in the background with traffic noise."
- **Sound event detection**: Classifying environmental sounds.
- **Music understanding**: Analyzing melody, tempo, and genre.

---

## 5. Key References

1.  **Radford et al. (2022)**: *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper).
2.  **Wang et al. (2023)**: *Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers* (VALL-E).
3.  **Rubenstein et al. (2023)**: *AudioPaLM: A Large Language Model That Can Speak and Listen*.
