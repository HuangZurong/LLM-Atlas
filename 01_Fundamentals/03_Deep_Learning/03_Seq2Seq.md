# Seq2Seq: Encoder-Decoder Architecture

*Prerequisite: [02_Sequence_Models.md](02_Sequence_Models.md).*

---

The Sequence-to-Sequence (Seq2Seq) model is the first general-purpose "input sequence → output sequence" framework. It gave rise to modern machine translation, text summarization, and dialogue systems.

## Contents

- [1. The Encoder-Decoder Framework](#1-the-encoder-decoder-framework)
- [2. The Information Bottleneck](#2-the-information-bottleneck)
- [3. Applications & Significance](#3-applications--significance)

## 1. The Encoder-Decoder Framework

Seq2Seq (Sutskever et al., 2014) chains two RNNs/LSTMs: an **Encoder** that compresses the input sequence into a vector, and a **Decoder** that generates the output sequence from that vector.

### Architecture

```
Encoder (source language):
  "I like cats" → [h1] → [h2] → [h3] → Context Vector (c)

Decoder (target language):
  Context Vector (c) → "我" → "喜欢" → "猫" → <EOS>
```

### Workflow

1. The **Encoder** reads the source sequence step by step, encoding information into a sequence of hidden states
2. The Encoder's final hidden state $h_T$ serves as the **Context Vector $c$**
3. The **Decoder** uses $c$ as its initial hidden state and generates target tokens autoregressively
4. Generation stops when the `<EOS>` (End of Sequence) token is produced

### Training: Teacher Forcing

During training, the Decoder receives the **ground truth target sequence** as input (rather than its own predictions):

```
Decoder input during training:   <BOS>  I    like   cats
Decoder target during training:   I    like   cats   <EOS>
```

- **Pros**: Accelerates convergence, avoids error accumulation
- **Cons**: Train-test distribution mismatch (Exposure Bias) — the model always sees correct input during training but may encounter its own errors during inference

## 2. The Information Bottleneck

The core problem of Seq2Seq: **the entire source sequence is compressed into a single fixed-dimension vector $c$**.

```
"The European Commission said on Thursday it disagreed with
 German advice to consumers to shun British lamb..."

      ↓ compressed entirely into

  [0.12, -0.34, 0.56, ...] (512 dimensions)
```

### The Problem

- The longer the source sentence, the more information the Context Vector loses
- Experiments show that basic Seq2Seq BLEU scores drop sharply when source sentences exceed 20 words
- Long-range correspondences (e.g., words at the beginning and end that relate to each other) are nearly impossible to preserve

### Solutions

This bottleneck directly led to two milestone improvements:

1. **Attention Mechanism**: Allows the Decoder to "look back" at all Encoder hidden states at each generation step, rather than relying solely on the final one
2. **Transformer**: Completely discards the RNN structure, using Self-Attention for global information exchange

These are covered in detail in [04_Transformer_Era](../04_Transformer_Era/).

## 3. Applications & Significance

### Classic Applications

| Task | Input | Output |
|:-----|:------|:-------|
| **Machine Translation** | Source language sentence | Target language sentence |
| **Text Summarization** | Long document | Condensed summary |
| **Dialogue Generation** | User message | System response |
| **Speech Recognition** | Audio feature sequence | Text sequence |

### Historical Significance

Seq2Seq established the paradigm of **end-to-end learning** in NLP — no more hand-designing intermediate steps (alignment tables, translation rules). A single neural network maps directly from input to output.

The Encoder-Decoder framework's influence persists today:
- **Transformer** itself is an Encoder-Decoder architecture (simply replacing RNN with Attention)
- **T5, BART** maintain the full Encoder-Decoder structure
- **GPT** series uses only the Decoder half (Decoder-only)

---

_The information bottleneck of Seq2Seq, combined with the sequential computation bottleneck of RNNs, jointly point to the next revolution in NLP — Attention and Transformer._

_Next: [Attention Mechanism](../04_Transformer_Era/01_Attention.md)_
