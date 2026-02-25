# Search and Recommendation

*Prerequisite: [../03_Deep_Learning/01_Word_Embeddings.md](../03_Deep_Learning/01_Word_Embeddings.md).*

---

NLP powers the understanding layer of search engines and recommendation systems — two of the highest-revenue AI applications in industry.

## 1. Industrial Systems — Search

| System | Company | Year | Venue | Description |
|:-------|:--------|:-----|:------|:------------|
| **BERT for Search** | Google | 2019 | Production | First integration of BERT into search ranking; impacts 1 in 10 queries; runs on Cloud TPUs |
| **MUM** | Google | 2021 | Production | 1000x more powerful than BERT; multimodal (text+images); handles complex queries like "I've hiked Mt. Fuji, what should I do differently for Mt. Hood?" |
| **Deep NLP for LinkedIn** | LinkedIn | 2021 | arXiv | BERT-based ranking for Job and Member search; open-sourced **DeText** framework for industrial ranking |
| **Semantic Product Search** | Amazon | 2019 | KDD | End-to-end semantic query-product matching; handles synonyms and behavioral associations (e.g., "warm clothes" → jackets) |
| **DPR (Dense Passage Retrieval)** | Meta | 2020 | EMNLP | Efficient dense retrieval using dual-encoders; foundation for modern RAG and vector-based search |
| **Que2Search** | Meta | 2021 | KDD | Query and product representation learning for Marketplace search; uses XLM-R to handle multilingual queries |

## 2. NLP's Role in Recommendation

| Component | NLP Role | Example |
|:----------|:---------|:--------|
| **Content Understanding** | Extracting topics, sentiment, and entities from items | **ByteDance (Toutiao)**: NLP understanding of news text for personalized feed |
| **User Interest Modeling** | Encoding user historical behavior (queries/clicks) into embeddings | **Pinterest (PinSage)**: Graph-based embeddings for content discovery |
| **Review Mining** | Summarizing pros/cons from millions of reviews | **Amazon**: "Customers frequently mention: battery life, camera quality" |
| **Cold-Start Handling** | Using item text to recommend new items with zero click data | **Spotify**: Using track metadata and audio descriptions to recommend new songs |

## 3. Key Technical Components

### 3.1 Query Understanding
In search, the query is often short and ambiguous.
- **Query Expansion**: Using synonyms or LLMs to expand "shoes" to "sneakers, footwear, boots".
- **Intent Detection**: Classification to determine if a query is Navigational, Informational, or Transactional.

### 3.2 Dense vs. Sparse Retrieval
- **Sparse (BM25/TF-IDF)**: Fast, keyword-exact match. Fails on "sofa" vs "couch".
- **Dense (Vector Search)**: Uses BERT/Bi-Encoders to map queries and documents to a vector space. Finds "sofa" when you search for "couch".
- **Hybrid Search**: Combining both for maximum recall and precision.

## 4. Engineering Insights: The Two-Tower Model

Most industrial search and recommendation systems use the **Two-Tower Architecture** for low-latency retrieval:

```
Query Tower (Encoder)         Document Tower (Encoder)
        ↓                             ↓
 Query Embedding (Vector)      Doc Embedding (Vector)
        ↘                             ↙
           Dot Product / Cosine Similarity
                        ↓
                 Top-K Candidates
```
The Document Tower can be pre-computed (indexed), while only the Query Tower runs in real-time, allowing search over billions of items in milliseconds.

## Key References

- Nayak, "[Understanding Searches Better than Ever Before (BERT for Search)](https://blog.google/products/search/search-language-understanding-bert/)", Google Blog 2019
- Guo et al., "[Deep Natural Language Processing for LinkedIn Search](https://arxiv.org/abs/2108.13300)", arXiv 2021
- Nigam et al., "[Semantic Product Search](https://arxiv.org/abs/1907.00937)", KDD 2019
- Karpukhin et al., "[Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)", EMNLP 2020
- Liu et al., "[Que2Search: Fast and Accurate Query and Product Understanding for Search at Facebook](https://arxiv.org/abs/2108.13112)", KDD 2021

---

_Next: [Content Moderation](./08_Content_Moderation.md) — Detecting and filtering harmful content at scale._
