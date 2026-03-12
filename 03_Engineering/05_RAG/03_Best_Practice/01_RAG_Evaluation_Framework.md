# RAG Evaluation Framework

*Prerequisite: [../01_Theory/02_Advanced_RAG.md](../01_Theory/02_Advanced_RAG.md).*

---

Evaluating RAG systems is fundamentally different from evaluating standalone LLMs. You must measure both the **Retrieval Quality** and the **Generation Quality** independently, then assess their interaction.

## 1. The RAGAS Framework

RAGAS (Retrieval Augmented Generation Assessment) is the industry standard for automated RAG evaluation.

### Core Metrics:

| Metric | What it Measures | Formula (Simplified) |
| :--- | :--- | :--- |
| **Faithfulness** | Does the answer only use information from the retrieved context? | % of claims in the answer that are supported by the context |
| **Answer Relevancy** | Does the answer address the original question? | Cosine similarity between the question and the generated answer |
| **Context Precision** | Are the retrieved chunks actually relevant? | % of retrieved chunks that contain the answer |
| **Context Recall** | Did we retrieve ALL the necessary information? | % of ground-truth facts that appear in the retrieved context |

### The Evaluation Matrix:

```
                    Retrieval Good    Retrieval Bad
                  ┌─────────────────┬─────────────────┐
Generation Good   │  ✅ Ideal        │  ⚠️ Lucky guess  │
                  ├─────────────────┼─────────────────┤
Generation Bad    │  ⚠️ Wasted       │  ❌ Total fail   │
                  │    context       │                 │
                  └─────────────────┴─────────────────┘
```

## 2. Building a Golden Dataset

Every production RAG system needs a **Golden Dataset** for regression testing.

### Structure:
```json
{
  "question": "What is the warranty period for Product X?",
  "ground_truth_answer": "2 years from date of purchase.",
  "ground_truth_contexts": ["doc_id_42", "doc_id_87"],
  "metadata": {"category": "warranty", "difficulty": "easy"}
}
```

### How to Build:
1. **Manual Curation**: Domain experts write 50-100 Q&A pairs covering edge cases.
2. **LLM-Assisted Generation**: Use GPT-4o to generate candidate Q&A pairs from your documents, then have humans verify.
3. **Production Mining**: Log real user queries and manually annotate the correct answers.

### Size Guidelines:
| Stage | Golden Set Size | Purpose |
| :--- | :--- | :--- |
| **Prototype** | 20-50 | Quick sanity check |
| **Pre-production** | 100-200 | Comprehensive coverage |
| **Production** | 500+ | Regression testing with statistical significance |

## 3. Automated Evaluation Pipeline

```
Code Change (Chunking/Embedding/Prompt)
    ↓
CI Pipeline Triggered
    ↓
Run Golden Dataset through RAG Pipeline
    ↓
Compute RAGAS Metrics
    ↓
Compare against Baseline (main branch)
    ↓
Pass/Fail Gate:
  - Faithfulness > 0.85
  - Context Precision > 0.75
  - Answer Relevancy > 0.80
```

## 4. Beyond RAGAS: Industrial Evaluation Dimensions

| Dimension | Metric | Tool |
| :--- | :--- | :--- |
| **Latency** | p50/p95 end-to-end response time | Custom logging / Langfuse |
| **Cost** | $ per query (embedding + LLM tokens) | Token counters |
| **Hallucination Rate** | % of answers containing unsupported claims | LLM-as-Judge |
| **Robustness** | Performance on adversarial/ambiguous queries | Manual test suite |
| **Freshness** | Can the system answer questions about recently ingested docs? | Timestamp-based tests |

## 5. Common Failure Modes & Diagnostics

| Symptom | Root Cause | Fix |
| :--- | :--- | :--- |
| High Faithfulness, Low Relevancy | Retrieved correct docs but answered the wrong question | Improve query transformation |
| Low Faithfulness, High Relevancy | Answer sounds right but hallucinates details | Strengthen grounding instructions in system prompt |
| Low Context Precision | Retrieving too many irrelevant chunks | Add reranking step or tighten metadata filters |
| Low Context Recall | Missing relevant documents entirely | Check chunking strategy, add HyDE or Multi-Query |
