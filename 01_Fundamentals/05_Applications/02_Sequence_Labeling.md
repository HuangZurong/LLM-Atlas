# Sequence Labeling

*Prerequisite: [01_Text_Classification.md](01_Text_Classification.md).*

---

**Task**: Assign a label to each token in a sequence — the foundation of structured information extraction and dialogue understanding.

## 1. Sub-tasks

| Task | Input | Output | Example |
|:-----|:------|:-------|:--------|
| **NER** | Sentence | Entity spans + types | "Tim Cook announced iPhone in Cupertino" → [Tim Cook: PER, iPhone: PROD, Cupertino: LOC] |
| **POS Tagging** | Sentence | Part-of-speech per token | "The/DET cat/NOUN sat/VERB" |
| **Slot Filling** | User utterance | Slot-value pairs | "Fly to Tokyo on Friday" → {dest: Tokyo, date: Friday} |
| **Chunking** | Sentence | Syntactic phrases | "[The big cat]NP [is sitting]VP [on the mat]PP" |
| **Semantic Role Labeling** | Sentence | Predicate-argument structure | "[Who]Agent [ate]Predicate [the apple]Theme" |

## 2. Technical Evolution

```
HMM (Generative, local dependencies)
    ↓
CRF (Discriminative, global dependencies)
    ↓
BiLSTM-CRF (Neural features + CRF transition layer)
    ↓
BERT-CRF / RoBERTa-CRF (Pre-trained representations)
    ↓
Instruction-tuning / In-context Learning (Zero-shot / Few-shot extraction)
```

## 3. Industrial Systems

| System | Company | Year | Venue | Description |
|:-------|:--------|:-----|:------|:------------|
| **Scaling up Open Tagging** | Alibaba | 2019 | ACL | Query-conditioned sequence labeling for product attribute extraction; scales to thousands of attributes across AliExpress |
| **TripleLearn NER** | Amazon | 2021 | AAAI | End-to-end NER for e-commerce search — improved F1 from 69.5 to 93.3 for brand/product type extraction |
| **CNER-UAV** | Meituan | 2024 | arXiv | Fine-grained Chinese address NER for UAV delivery; 12K labeled samples across 5 address categories |
| **Comprehend Medical** | Amazon (AWS) | 2018 | Production | HIPAA-eligible NER service for clinical text; extracts entities across medications, conditions, and PHI |
| **KB-NER (DAMO-NLP)** | Alibaba | 2022 | SemEval | Won Best System Paper; knowledge-based multilingual NER with external knowledge retrieval |
| **Span-BERT for Slot Filling** | Google/Meta | 2020 | Production | Span-based extraction for digital assistants; superior performance in multi-turn slot carryover |

## 4. The BERT-CRF Pattern

In production, combining **BERT** (for contextual feature extraction) with a **CRF** (Conditional Random Field) layer remains a gold standard for NER because it ensures **global label consistency**.

```
Input tokens  →  BERT Encoder  →  Token representations  →  CRF Layer  →  Label sequence
                                                              ↑
                                                 Ensures valid transitions
                                                 (e.g., B-PER → I-PER is valid,
                                                  but B-PER → I-LOC is penalized)
```

## 5. Production Challenges

### 5.1 The OOV (Out-of-Vocabulary) Problem
Industrial NER systems frequently encounter new product names, brands, or medical terms. Solutions include:
- **Subword Tokenization**: Using BPE or WordPiece to handle rare words.
- **Gazetteer Integration**: Augmenting neural models with dictionaries or knowledge base lookups.

### 5.2 Annotation Bottleneck
Labeling tokens is much slower than labeling whole sentences. Industrial teams use:
- **Active Learning**: Only annotating tokens the model is most uncertain about.
- **Weak Supervision**: Using rules or legacy dictionaries to auto-label data for initial training.

## Key References

- Zheng et al., "[OpenTag: Open Attribute Value Extraction from Product Profiles](https://arxiv.org/abs/1806.01264)", KDD 2018
- Huang et al., "[Scaling up Open Tagging from Tens to Thousands](https://aclanthology.org/P19-1505/)", ACL 2019
- Fetahu et al., "[TripleLearn End-to-End NER for eCommerce](https://ojs.aaai.org/index.php/AAAI/article/view/21459)", AAAI 2021
- DAMO-NLP, "[KB-NER: a Knowledge-based System for Multilingual Complex Named Entity Recognition](https://arxiv.org/abs/2203.00545)", SemEval 2022 Best System Paper
- Joshi et al., "[SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529)", TACL 2020

---

_Next: [Information Extraction](./03_Information_Extraction.md) — Extracting structured knowledge from unstructured text._
