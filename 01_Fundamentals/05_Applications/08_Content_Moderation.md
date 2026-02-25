# Content Moderation & Trust and Safety

*Prerequisite: [01_Text_Classification.md](01_Text_Classification.md).*

---

**Task**: Detect and filter harmful, illegal, or policy-violating content at scale — the defensive backbone of the internet.

## 1. Sub-tasks

| Task | Description | Industrial Example |
|:-----|:-----------|:-------------------|
| **Toxicity Detection** | Identifying hate speech, harassment, threats | Perspective API, Llama Guard |
| **Misinformation** | Detecting fake news or manipulated facts | Fact-checking pipelines at Meta/Twitter |
| **NSFW Detection** | Adult content, gore, inappropriate imagery | Rosetta (Text-in-Image) |
| **PII Detection** | Sensitive data like emails, phone numbers, SSNs | Data Loss Prevention (DLP) tools |
| **Spam Filtering** | High-volume unwanted commercial content | Gmail RETVec, LinkedIn Viral Spam |

## 2. Industrial Systems

| System | Company | Year | Venue | Description |
|:-------|:--------|:-----|:------|:------------|
| **Perspective API** | Google (Jigsaw) | 2022 | arXiv | Multilingual character-level Transformer for toxicity detection; serves 2B+ requests/day |
| **Llama Guard** | Meta | 2023 | arXiv | 7B-parameter LLM-based safety classifier; provides customizable taxonomy for AI safety |
| **Rosetta** | Meta | 2018 | KDD | OCR system for text-in-image moderation; processes 1B+ images/video frames daily |
| **RETVec** | Google (Gmail) | 2023 | NeurIPS | Distilled, robust text vectorizer for spam detection; significantly reduced TPU usage |
| **HateSpeech XLM-R** | Meta | 2021 | arXiv | Massive cross-lingual pre-trained model for detecting policy violations in 100+ languages |
| **Presidio** | Microsoft | 2020 | Open Source | Production-ready SDK for PII detection and anonymization using NER and pattern matching |

## 3. Engineering Reality: The Production Stack

Moderation at scale (billions of items/day) requires a multi-layered defense:

1. **Heuristic Layer (Fast)**: Regex, blocklists, and hash-matching for known illegal content (e.g., CSAM hashes).
2. **Specialized Models (Efficient)**: Distilled BERT or character-level models (Perspective) for < 50ms latency.
3. **LLM/Human-in-the-Loop (Precise)**: Routing the most ambiguous cases to a high-capacity LLM or human moderator.

## 4. Key Challenges

### 4.1 Adversarial Robustness
Bad actors use "leetspeak" (e.g., `h4te`) or emojis to bypass filters.
- **Solution**: Character-level models or visual-text models (like Rosetta) that "see" the text as a human would.

### 4.2 Contextual Understanding
The same sentence can be toxic or harmless depending on context (e.g., reclaimed slurs vs. attacks).
- **Solution**: Multi-modal and context-aware models that look at the surrounding conversation or user history.

### 4.3 Multilingual & Cultural Nuance
What is offensive in one culture may be acceptable in another.
- **Solution**: Language-specific fine-tuning and cross-lingual pre-training (XLM-R).

## 5. Why Specialists Win in Moderation

- **Latency**: Platforms like YouTube or Facebook cannot wait 2 seconds for an LLM to generate a safety label for every comment.
- **Cost**: A $0.01 API call per comment is impossible at a scale of 100 billion comments/day.
- **Coverage**: Specialized models can be tiny (e.g., < 50MB) and deployed on edge devices or in high-efficiency CPU environments.

## Key References

- Lees et al., "[A New Generation of Perspective API: Efficient Multilingual Character-level Transformers](https://arxiv.org/abs/2202.11176)", arXiv 2022
- Inan et al., "[Llama Guard: LLM-based Input-Output Safeguard](https://arxiv.org/abs/2312.06674)", arXiv 2023
- Borisyuk et al., "[Rosetta: Large Scale System for Text Detection and Recognition in Images](https://dl.acm.org/doi/10.1145/3219819.3219861)", KDD 2018
- Botha et al., "[RETVec: Resilient & Efficient Text Vectorizer](https://arxiv.org/abs/2302.09207)", NeurIPS 2023
- Conneau et al., "[Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116)", ACL 2020

---

_Next: [The LLM Disruption Map](./09_LLM_Disruption_Map.md) — Where LLMs replace specialized models, and where they don't._
