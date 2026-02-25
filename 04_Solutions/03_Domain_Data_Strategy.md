# Domain Data Strategy: From Raw Sources to Training-Ready Corpus

*Prerequisite: [01_Technology_Selection.md](01_Technology_Selection.md), [../02_Scientist/02_Dataset/](../02_Scientist/02_Dataset/).*

---

This document addresses the practical question: given a specific domain (e.g., infrastructure construction-operations), how do you plan and execute the data pipeline that feeds your LLM system? This is not about data preprocessing techniques (covered in 02_Scientist/02_Dataset) but about strategic decisions.

## 1. Data Needs Assessment

Before collecting a single document, answer these questions:

### 1.1 What Will the Data Be Used For?

| Purpose | Data Type Needed | Volume Estimate | Quality Bar |
| :--- | :--- | :--- | :--- |
| **RAG knowledge base** | Raw documents, structured records | Hundreds to thousands of documents | Medium (retrieval tolerates noise) |
| **Fine-tuning for domain adaptation** | Domain text corpus | Millions of tokens | Medium-high (diverse, representative) |
| **Fine-tuning for task-specific behavior** | Input-output pairs (instruction format) | 1K-50K examples | High (must be accurate and consistent) |
| **Evaluation** | Gold-standard Q&A pairs with verified answers | 200-1000 examples | Very high (expert-validated) |
| **Knowledge graph construction** | Entity-relation triples | Thousands of triples | High (must be factually correct) |

The same raw source can serve multiple purposes, but the processing pipeline differs for each.

### 1.2 The Data Pyramid

```
                    /\
                   /  \        Evaluation data (expert-curated, 200-1K)
                  /    \
                 /------\      Instruction data (structured pairs, 1K-50K)
                /        \
               /----------\    Domain corpus (cleaned text, millions of tokens)
              /            \
             /--------------\  Raw sources (everything you can get, uncleaned)
```

Each layer is smaller but higher quality. You build from bottom to top.

## 2. Source Identification and Collection

### 2.1 Source Categories

For a domain like infrastructure construction-operations:

**Publicly available**:
- Academic papers (CNKI, Google Scholar, IEEE Xplore)
- Industry standards and regulations (national/international)
- Government publications, policy documents
- Open textbooks, technical manuals
- Patent databases

**Semi-public** (requires access/agreements):
- Industry association reports
- Conference proceedings
- Consulting firm white papers
- Trade publications

**Private/proprietary** (requires partnerships):
- Project documentation (plans, schedules, change orders)
- Operational logs, maintenance records
- Internal reports, lessons learned
- Expert interview transcripts
- Meeting minutes, decision records

### 2.2 Collection Prioritization

Not all sources are equally valuable. Prioritize by:

1. **Relevance density**: How much of the document is actually about your domain? A 200-page textbook chapter is better than 200 pages of tangentially related news articles.
2. **Knowledge uniqueness**: Does this source contain knowledge the base model already has? General knowledge about "project management" is already in GPT-4. Domain-specific knowledge about "airport runway construction phasing" is not.
3. **Structural quality**: Well-structured documents (with headings, tables, clear sections) are easier to process and produce better training data.
4. **Recency**: For rapidly evolving domains, prioritize recent sources. For foundational knowledge, older authoritative texts are fine.

### 2.3 Legal and Ethical Considerations

- Copyright status of each source
- Data licensing terms (especially for commercial use)
- Personal data / PII in project documents (must be anonymized)
- Confidentiality agreements with data providers
- Institutional review board (IRB) requirements for interview data

## 3. Processing Pipeline Design

### 3.1 For RAG Knowledge Base

```
Raw documents → Format conversion → Cleaning → Chunking → Embedding → Vector DB
                (PDF/DOCX→text)    (noise     (semantic    (domain-aware
                                    removal)   boundaries)  or general)
```

Key decisions:
- **Chunk strategy**: Fixed-size (simple but breaks context) vs semantic (respects section boundaries) vs hierarchical (parent-child chunks for multi-granularity retrieval)
- **Metadata preservation**: Keep source, date, section title, document type as filterable metadata
- **Embedding model**: General-purpose (BGE, E5) vs domain-fine-tuned. Start general, fine-tune later if retrieval quality is insufficient.

### 3.2 For Fine-tuning Corpus (Pre-training/CPT)

```
Raw documents → Format conversion → Cleaning → Deduplication → Quality filtering → De-contamination → Tokenization
                                                                 (perplexity,
                                                                  language ID,
                                                                  content filters)
```

**Industrial Cleaning Pipeline (The "Big Science" Pattern):**

1. **Rule-based Filters**:
   - Language identification (fastText).
   - Stop-word ratio (filter out gibberish).
   - Symbol-to-word ratio (filter out code or math-heavy noise if not desired).
   - "Boilerplate" removal (headers, footers, navigation menus).
2. **Model-based Filtering**:
   - Use a lightweight model (e.g., fastText or a small BERT) trained on "high-quality" vs "low-quality" samples to score documents.
   - Perplexity filtering: Use a small LLM (e.g., Qwen-0.5B) to calculate perplexity; remove extremely high (gibberish) or extremely low (repetitive boilerplate) outliers.
3. **Fuzzy Deduplication**:
   - MinHash + LSH at the document level.
   - Semantic deduplication for instruction data (clustering embeddings).
4. **De-contamination**:
   - **CRITICAL**: Use n-gram overlap check (typically 13-gram) between your training corpus and all public benchmarks (MMLU, CMMLU, C-Eval, GSM8K) plus your internal evaluation set. Remove any training sample that overlaps.

### 3.3 For Instruction Data (SFT & Alignment)

Instruction data is the "intelligence" layer. When domain data is scarce, we use **Synthetic Data Engineering**.

**Synthetic Data Generation Pipelines:**

| Method | Description | Best For |
| :--- | :--- | :--- |
| **Self-Instruct** | Using an LLM to generate new instructions from seed tasks. | Expanding task variety. |
| **Evol-Instruct** | Iteratively increasing instruction complexity (adding constraints, steps, or reasoning depth). | Improving model's reasoning capability. |
| **Magpie** | Extracting instruction-response pairs from raw model "self-conversations" without explicit prompts. | High-volume, low-cost diversity. |
| **Knowledge-to-Instruction (K2I)** | Converting technical manuals/tables into "Q: [Question] A: [Answer based on Document]" format. | Injecting domain facts. |
| **Back-translation** | Given a document, generate a question that would be answered by that document. | Grounded RAG-style SFT. |

**The "Agent-in-the-loop" Generation Pattern:**

1. **Generator Agent**: Creates candidates (e.g., using GPT-4o).
2. **Critic Agent**: Reviews candidates for logical flaws or domain inaccuracies.
3. **Refiner Agent**: Fixes identified issues.
4. **Expert Audit**: Human-in-the-loop validation of a 5% random sample.

### 3.4 Instruction Data Format

For chat-style fine-tuning, structure data as conversations:

```json
{
  "messages": [
    {"role": "system", "content": "You are a construction operations expert..."},
    {"role": "user", "content": "What are the key risks during the transition from construction to operations phase?"},
    {"role": "assistant", "content": "The construction-to-operations transition involves several critical risks: 1) ..."}
  ]
}
```

For different task types, vary the instruction format:
- **Knowledge Q&A**: Direct question → detailed answer
- **Document analysis**: "Given this report excerpt: [context]. Question: ..." → analysis
- **Decision support**: Scenario description → recommended actions with reasoning
- **Report generation**: Brief inputs → structured report output

## 4. Data Quality Assurance

### 4.1 Automated Checks

- Language detection (filter out wrong-language content)
- Encoding validation (detect and fix garbled text)
- Length filtering (too short = low information, too long = likely noise)
- Perplexity scoring (flag outliers for manual review)
- PII detection (names, phone numbers, addresses)
- Duplicate detection (exact and near-duplicate)

### 4.2 Human Review Protocol

For instruction data, establish a review protocol:

1. **Factual accuracy**: Is the answer correct? (Requires domain expert)
2. **Completeness**: Does the answer address all aspects of the question?
3. **Consistency**: Does this answer contradict other answers in the dataset?
4. **Tone and style**: Does it match the desired output style?
5. **Harmful content**: Any biased, misleading, or dangerous advice?

Target: Review at least 10-20% of LLM-generated instruction data. Review 100% of evaluation data.

### 4.3 Iterative Refinement

Data quality is not a one-time effort. After initial model training:

1. Test the model on held-out evaluation set
2. Identify failure categories (wrong facts, missing knowledge, wrong format)
3. Create targeted training data to address each failure category
4. Retrain and re-evaluate
5. Repeat

This "data flywheel" is often more effective than scaling up data volume blindly.

## 5. Data Volume Guidelines

| Component | Minimum Viable | Recommended | Notes |
| :--- | :--- | :--- | :--- |
| **RAG corpus** | 100 documents | 1000+ documents | More is generally better for coverage |
| **Fine-tuning corpus** (continued pre-training) | 10M tokens | 100M+ tokens | Domain text for vocabulary/concept adaptation |
| **Instruction data** (SFT) | 1K examples | 5K-20K examples | Quality matters more than quantity |
| **Evaluation set** | 100 examples | 500+ examples | Must be expert-validated, never used for training |

These are rough guidelines. The right volume depends on domain complexity, task difficulty, and base model capability.

## 6. Common Pitfalls

1. **Collecting everything, cleaning nothing**: Raw data volume is meaningless. 10K clean examples beat 100K noisy ones.
2. **Ignoring domain balance**: If 80% of your data is about "safety regulations" and 20% about "cost management," the model will be great at safety and terrible at cost.
3. **Training on evaluation data**: Accidentally including test examples in training data. Use strict data splits and checksums.
4. **Assuming OCR output is clean**: PDF extraction and OCR introduce significant noise. Always validate a sample.
5. **Neglecting metadata**: Losing track of which data came from where makes debugging and updating impossible.
6. **One-shot data collection**: Treating data as a one-time task rather than an ongoing process. Domain knowledge evolves; your data pipeline should too.

---

## Key References

1. **Gururangan et al. (2020)**: *Don't Stop Pretraining: Adapt Pretrained Language Models to Domains and Tasks*.
