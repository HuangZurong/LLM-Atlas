# Machine Translation

*Prerequisite: [../04_Transformer_Era/02_Transformer.md](../04_Transformer_Era/02_Transformer.md).*

---

**Task**: Convert text from one language to another — one of the oldest and most commercially significant NLP tasks.

## 1. Sub-tasks

| Task | Description | Industrial Use Case |
|:-----|:-----------|:-------------------|
| **Neural MT (NMT)** | General-purpose text translation | Google Translate, DeepL |
| **Simultaneous MT** | Real-time translation with low latency | Live stream subtitling, Baidu STACL |
| **Terminology-Aware MT** | Translation preserving domain-specific terms | Legal/Medical document translation |
| **Quality Estimation (QE)** | Predicting translation quality without reference | Routing low-quality translations to human editors |

## 2. Technical Evolution

```
Rule-based (1950s: Word-to-word, grammar rules)
    ↓
Statistical MT (1990s: Phrase-based, alignment models, Moses)
    ↓
Neural Seq2Seq (2014: GRU/LSTM + Attention)
    ↓
Transformer (2017: Parallel training, self-attention)
    ↓
Massively Multilingual (2022: Meta NLLB, single model for 200+ languages)
    ↓
LLM MT (2023+: GPT-4/Claude, few-shot with terminology/style guides)
```

## 3. Industrial Systems

| System | Company | Year | Venue | Description |
|:-------|:--------|:-----|:------|:------------|
| **GNMT** | Google | 2016 | arXiv | 8-layer encoder-decoder LSTM with attention; reduced translation errors by 60% vs. phrase-based; serves 500M+ users |
| **NLLB-200** | Meta | 2022 | Nature 2024 | 200-language MT using Sparsely Gated MoE; serves 25B+ translations/day across Meta platforms |
| **STACL** | Baidu | 2019 | ACL | First simultaneous translation system with anticipation; "wait-k" model for controllable latency-quality tradeoff |
| **DeepL Translator** | DeepL | 2017+ | Proprietary | Evolved from CNN to Transformer + LLM; known for superior European language quality and stylistic nuance |
| **WMT Competition** | Industry-Wide | Annual | WMT | The "Olympics" of MT where companies like Microsoft, Baidu, and Alibaba compete on SOTA benchmarks |
| **Comet (Quality Estimation)** | Unbabel | 2020 | EMNLP | Production-ready MT evaluation model that predicts human scores (MQM) with high correlation |

## 4. Production Reality

### 4.1 Quality Estimation (QE) & Human-in-the-Loop
Industrial MT isn't just about the model; it's about the **pipeline**. For high-stakes content (e.g., medical device manuals):
1. **Model Translation**: Fast Transformer-based NMT.
2. **QE Filter**: A model (like COMET or BERTScore) predicts the confidence.
3. **Routing**: High-confidence translations go to production; low-confidence ones go to **Human Post-Editing (PE)**.

### 4.2 Domain Adaptation
A general MT model often fails on niche terminology (e.g., "Apple" as a company vs. fruit).
- **Solution**: Terminology injection via constrained decoding or fine-tuning on domain-specific translation memories (TM).

### 4.3 Multilingual Challenges
Serving 100+ separate models (one per language pair) is an engineering nightmare.
- **Solution**: Multilingual NMT (MNMT) where a single model handles multiple directions, often showing **transfer learning** benefits from high-resource (EN-FR) to low-resource (EN-Swahili) pairs.

## 5. Current State: LLM vs. Specialized MT

- **High-resource pairs**: LLMs (GPT-4) match or exceed specialized MT, especially in capturing "style" or "tone".
- **Low-resource languages**: Specialized multilingual models (NLLB) still win due to better coverage of rare tokens.
- **Cost**: For massive document localization, specialized Transformers are still significantly cheaper per token.

## Key References

- Wu et al., "[Google's Neural Machine Translation System](https://arxiv.org/abs/1609.08144)", arXiv 2016
- NLLB Team, "[No Language Left Behind](https://arxiv.org/abs/2207.04672)", Nature 2024
- Ma et al., "[STACL: Simultaneous Translation with Anticipation and Controllable Latency](https://arxiv.org/abs/1810.08398)", ACL 2019
- Rei et al., "[COMET: Efficient and High Quality Machine Translation Evaluation](https://arxiv.org/abs/2009.09025)", EMNLP 2020
- Vaswani et al., "[Attention Is All You Need](https://arxiv.org/abs/1706.03762)", NeurIPS 2017

---

_Next: [Text Summarization](./05_Text_Summarization.md) — Condensing documents while preserving key information._
