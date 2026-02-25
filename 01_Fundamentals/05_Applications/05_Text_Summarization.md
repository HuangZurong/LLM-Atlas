# Text Summarization

*Prerequisite: [../04_Transformer_Era/03_Pre_Training_Paradigms.md](../04_Transformer_Era/03_Pre_Training_Paradigms.md).*

---

**Task**: Condense long documents into shorter versions while preserving key information — critical for news aggregation, legal review, and content consumption.

## 1. Sub-tasks

| Task | Description | Industrial Use Case |
|:-----|:-----------|:-------------------|
| **Single-Document** | Summarize one article/email | Gmail Smart Summary, News apps |
| **Multi-Document** | Aggregate info from multiple sources | Topic-based news clusters (e.g., Google News) |
| **Query-Focused** | Summarize relevant to a user query | Search engine snippets, Customer support |
| **Dialogue Summarization** | Summarize meetings or chat logs | Zoom/Teams meeting notes, CRM summaries |

## 2. Technical Evolution

```
Extractive (Unsupervised)
    ↓
Neural Extractive (RNN/CNN classification of sentences)
    ↓
Seq2Seq Abstractive (LSTM + Attention, copy-mechanism)
    ↓
Pre-trained Abstractive (BART / T5 / PEGASUS - Denoising objectives)
    ↓
LLM-based (GPT-4 / Claude - Zero-shot abstractive, multi-aspect summarization)
```

## 3. Industrial Systems

| System | Company | Year | Venue | Description |
|:-------|:--------|:-----|:------|:------------|
| **PEGASUS** | Google | 2020 | ICML | Pre-training with Gap-Sentence Generation; SOTA for abstractive summarization; captures core meaning with few examples |
| **Google Docs AI** | Google | 2022 | Product | Production summarization using distilled PEGASUS; optimized for real-time document summary generation |
| **Deep Reinforced Sum** | Salesforce | 2018 | ICLR | First application of Reinforcement Learning to summarization to avoid "exposure bias" in generation |
| **FactCC** | Salesforce | 2020 | EMNLP | Model to verify factual consistency; addresses the "hallucination" problem where models generate facts not in the source |
| **SummaRuNNer** | IBM | 2017 | AAAI | RNN-based extractive summarization system; efficient and interpretable for enterprise document processing |
| **Meeting Summarizer** | Microsoft | 2021 | Blog/KDD | Production system for Teams; handles speaker identification and action item extraction using multi-task learning |

## 4. Production Challenges

### 4.1 The Hallucination Problem
Abstractive models often "hallucinate" facts (e.g., changing dates or names).
- **Solution**: Industrial systems often use an **Extractive-Abstractive Hybrid** or include a **Fact-Checking layer** (like FactCC) to flag inconsistent summaries.

### 4.2 Multi-Document Aggregation
When source documents overlap or conflict (e.g., multiple news reports on the same event).
- **Solution**: Clustering-based approaches where sentences are grouped by topic before a summarization model selects the most representative and unique information.

### 4.3 Evaluation Beyond ROUGE
Standard metrics like ROUGE only measure n-gram overlap.
- **Solution**: Industry increasingly uses **LLM-as-a-Judge** (G-Eval) or semantic metrics like **BERTScore** to evaluate if the "meaning" is preserved.

## 5. Why LLMs are Winning Summarization

LLMs have largely "solved" general summarization because:
- **World Knowledge**: They understand context better (e.g., knowing who a person is helps determine their importance).
- **Controllability**: Users can specify "Summary in 3 bullet points" or "Professional tone" via prompting.
- **RAG Integration**: LLMs can summarize retrieved context from thousands of pages, effectively doing large-scale multi-document summarization.

## Key References

- Zhang et al., "[PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://arxiv.org/abs/1912.08777)", ICML 2020
- Paulus et al., "[A Deep Reinforced Model for Abstractive Summarization](https://arxiv.org/abs/1705.04304)", ICLR 2018
- Kryscinski et al., "[Evaluating the Factual Consistency of Abstractive Text Summarization](https://arxiv.org/abs/1910.12840)", EMNLP 2020
- Nallapati et al., "[SummaRuNNer: A Recurrent Neural Network Based Sequence Model for Extractive Summarization of Documents](https://arxiv.org/abs/1611.04230)", AAAI 2017
- Lewis et al., "[BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)", ACL 2020

---

_Next: [Dialogue Systems](./06_Dialogue_Systems.md) — Enabling natural language conversation between humans and machines._
