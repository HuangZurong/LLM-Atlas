# Vertical Scenario Templates: Industry-Specific LLM Architectures

*Prerequisite: [01_Technology_Selection.md](01_Technology_Selection.md) through [08_Agent_Workflow_Design.md](08_Agent_Workflow_Design.md).*

---

This document provides architectural blueprints for common industry vertical scenarios, moving from general LLM patterns to concrete, business-aligned solutions.

## 1. Professional Services: Legal & Compliance "Reviewer"

**Scenario**: Automating the review of 1000-page contracts against internal compliance standards.

### 1.1 Architecture Blueprint (Hybrid GraphRAG)
1. **Hierarchical Indexing**: Parse contracts into Clause -> Article -> Paragraph.
2. **Compliance Knowledge Graph**: Entities (Counterparty, Jurisdiction) + Rules (Force Majeure, Liability Caps).
3. **Retrieval**: Semantic search for clauses + Graph traversal to find conflicting rules.
4. **Agentic Logic**:
   - Agent A: Extracts key terms.
   - Agent B: Compares terms against the Compliance DB.
   - Agent C: Generates a risk report with citations.

### 1.2 Key Differentiator
- **Precision > Recall**: A missed clause is a liability.
- **Evidence-Based**: Every claim must link to a specific PDF page/coordinate.

---

## 2. Finance: Research & Portfolio "Analyst"

**Scenario**: Generating real-time investment summaries from earnings calls, news, and market data.

### 2.1 Architecture Blueprint (Multi-Source Agentic RAG)
1. **Dynamic Tools**: SQL agent (for market prices) + Vector retriever (for analyst reports) + Web search (for latest news).
2. **Contextual Window Management**: Summarize transcript sub-sections (Map-Reduce) before final synthesis.
3. **Guardrails**: PII masking of client names and quantitative validation (Self-Correction if numbers in summary don't match SQL output).

### 2.2 Key Differentiator
- **Freshness**: Information expires in minutes.
- **Numerical Integrity**: Hallucinating a 5.2% yield as 52% is catastrophic.

---

## 3. Manufacturing & Engineering: O&M "Companion"

**Scenario**: Troubleshooting complex equipment using technical manuals and sensor telemetry.

### 3.1 Architecture Blueprint (Dual-LLM RAG)
1. **Input Sanitization**: Block prompt injection or irrelevant chat.
2. **Knowledge Retrieval**: RAG over manuals (specialized in tables and diagrams).
3. **Telemetry Integration**: Inject current error codes and sensor values into the prompt.
4. **Output Verification**: Llama Guard filters out unsafe operational advice.

### 3.2 Key Differentiator
- **Modalities**: Must handle CAD drawings and technical tables (using Layout-aware OCR/Multimodal LLMs).
- **Physical Safety**: Instructions must be verified against safety protocols.

---

## 4. Healthcare: Clinical Decision Support "Assistant"

**Scenario**: Summarizing patient history and suggesting potential treatments based on medical guidelines.

### 3.1 Architecture Blueprint (HIPAA-Compliant Private Cloud)
1. **PII Redaction**: Automatic masking of Patient ID, Name, and DOB before model processing.
2. **Medical Knowledge Base**: RAG over UpToDate, PubMed, and internal clinical pathways.
3. **Expert-in-the-Loop**: Model suggests, Doctor approves/edits (HITL).
4. **Audit Trail**: Log of every source used for a recommendation.

### 3.2 Key Differentiator
- **Factuality**: Zero tolerance for "hallucinated" drugs or dosages.
- **Explainability**: "Why did you suggest this treatment?" must be answerable via citations.

---

## 5. Summary Matrix for Verticals

| Vertical | Priority | Core Tech | Safety Bar |
| :--- | :--- | :--- | :--- |
| **Legal** | Precision | GraphRAG / Citations | High (Legal Risk) |
| **Finance** | Freshness | Tool-use / SQL Agent | High (Financial Loss) |
| **Engineering** | Structure | Multimodal / OCR | Critical (Physical Harm) |
| **Medical** | Privacy | PII Masking / SFT | Critical (Life/Death) |
