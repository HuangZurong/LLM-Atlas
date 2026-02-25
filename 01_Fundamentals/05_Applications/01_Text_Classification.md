# Text Classification

*Prerequisite: [../04_Transformer_Era/03_Pre_Training_Paradigms.md](../04_Transformer_Era/03_Pre_Training_Paradigms.md).*

---

**Task**: Assign a predefined label (or set of labels) to a piece of text — the most fundamental NLU task in production.

## 1. Sub-tasks

| Task | Input | Output | Example |
|:-----|:------|:-------|:--------|
| **Sentiment Analysis** | Review / comment | Positive / Negative / Neutral | "This phone is amazing" → Positive |
| **Intent Recognition** | User utterance | Intent label | "Book a flight to Tokyo" → `book_flight` |
| **Topic Classification** | Document | Topic label(s) | News article → Sports / Politics / Tech |
| **Spam/Fraud Detection** | Email / Transaction | Spam / Ham (Normal) | "Winner of $1M!" → Spam |
| **Query Categorization** | Search query | Domain / Vertical | "best sushi near me" → Food/Restaurants |

## 2. Technical Evolution

```
Rule-based (Regex, Keywords)
    ↓
Statistical (Naive Bayes / SVM + TF-IDF)
    ↓
Deep Learning (TextCNN / BiLSTM / Attention)
    ↓
Pre-trained Contextual (BERT / RoBERTa / XLNet Fine-tuning)
    ↓
LLM Era (Few-shot Prompting / Instruction Tuning)
```

## 3. Industrial Systems

| System | Company | Year | Venue | Description |
|:-------|:--------|:-----|:------|:------------|
| **RETVec** | Google (Gmail) | 2023 | NeurIPS | Resilient text vectorizer for spam detection — improved catch rate by 38%, robust against adversarial text manipulations |
| **Alexa Teacher Model** | Amazon | 2022 | arXiv | 9.3B-parameter encoder pre-trained and distilled for Alexa NLU — +3.86% intent classification, +7.01% slot filling |
| **ERNIE 3.0** | Baidu | 2021 | arXiv | 10B-parameter knowledge-enhanced model; SOTA on 45+ downstream tasks including sentiment and news classification |
| **Viral Spam Detection** | LinkedIn | 2023 | Eng. Blog | Production system using Boosted Trees combining member behavior, content features, and interaction patterns |
| **MT5-NLU** | Google | 2021 | arXiv | Multilingual T5 for intent recognition across 100+ languages; handles zero-shot cross-lingual transfer for Google Assistant |
| **HateSpeech Detection** | Meta | 2021 | Blog | XLM-R based multilingual classification serving 2B+ users; reduced prevalence of hate speech by 50% through improved NLU |

## 4. Production Insights

### 4.1 Latency vs. Accuracy
In high-throughput environments like Gmail or Facebook, the latency budget for classification is often **< 20ms**. While LLMs provide higher zero-shot accuracy, production systems usually rely on:
- **Feature distillation**: Knowledge distilled from a large LLM/BERT into a fast BiLSTM or small Transformer.
- **Hierarchical classification**: Fast "triage" models (Linear/SVM) to filter easy cases, followed by expensive models for ambiguous ones.

### 4.2 Handling Domain Shift
Industrial data is non-stationary. Systems must include:
- **Active Learning**: Periodically sampling low-confidence predictions for human review.
- **Concept Drift Monitoring**: Tracking changes in label distributions (e.g., new spam patterns or user intents).

## 5. Why Specialized Models Still Matter

- **Latency**: Autoregressive generation (LLM) is orders of magnitude slower than single-pass classification (BERT).
- **Cost**: Serving billions of classifications via LLM APIs is economically non-viable.
- **Deterministic Output**: Fine-tuned classifiers provide a fixed label set, avoiding "hallucinated" or conversational labels.

## Key References

- Botha et al., "[RETVec: Resilient & Efficient Text Vectorizer](https://arxiv.org/abs/2302.09207)", NeurIPS 2023
- Soltan et al., "[AlexaTM 20B: Few-Shot Learning Using a Large-Scale Multilingual Seq2Seq Model](https://arxiv.org/abs/2208.01448)", arXiv 2022
- Sun et al., "[ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation](https://arxiv.org/abs/2107.02137)", arXiv 2021
- Xue et al., "[mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer](https://arxiv.org/abs/2010.11934)", NAACL 2021
- Goyal et al., "[XLM-R: Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116)", ACL 2020

---

_Next: [Sequence Labeling](./02_Sequence_Labeling.md) — Token-level labeling for NER, POS tagging, and slot filling._
