# Data Safety & PII Management: Governance at Scale

*Prerequisite: [01_Pre_Training_Data_at_Scale.md](01_Pre_Training_Data_at_Scale.md). Covers Microsoft Presidio, Llama Guard, and GDPR compliance for AI.*

---

## 1. PII Scrubbing (Privacy Management)

Personally Identifiable Information (PII) must be removed to prevent the model from "leaking" user secrets (Emails, SSNs, Phone numbers).

### 1.1 Multi-Layered Defense
1.  **Regex Scrubbing (Fast)**: Catching standard patterns like emails, IP addresses, and credit card numbers.
2.  **Transformer-based NER (Accurate)**: Using models like `BERT-base-NER` to detect Names, Locations, and Organizations that don't follow rigid patterns. **Microsoft Presidio** is an open-source orchestration framework that combines regex + NER + context-aware rules — it is not itself a model, but a pipeline that coordinates multiple detectors.
3.  **Contextual Masking**: Instead of deleting, we replace PII with tags: `My name is <NAME>` or `Contact me at <EMAIL>`.

---

## 2. Safety & Toxicity Filtering

### 2.1 Adversarial Toxicity Classifiers
Using a dedicated "Safety Model" (e.g., **Llama-Guard** or **ToxicChat**) to score every training sample.
- **Thresholding**: Discarding samples with a toxicity score $> 0.5$.
- **Adversarial Training Data**: Specifically including "Harmful Intent + Refusal" pairs to teach the model where the boundaries are.

---

## 3. Data Contamination & Benchmarks

### 3.1 N-Gram Overlap Check
The gold standard for scientific integrity. Before training, we scan our 15T tokens against all known benchmarks (MMLU, HumanEval, etc.).
- **Window**: Typically a 13-gram or 15-gram window.
- **Action**: If a document contains a benchmark question/answer, it is either deleted or down-weighted to zero.

---

## 4. Jailbreak Defense Construction

Modern datasets include **Jailbreak attempts** paired with **safe refusals**.
- **DAN (Do Anything Now) prompts**: Collecting variations of jailbreak attempts.
- **Principle-based Refusal**: Teaching the model to refuse based on specific safety guidelines (e.g., "I cannot provide medical advice").

---

## 5. Key Tools & Libraries

- **Microsoft Presidio**: The industry standard for PII detection.
- **Llama-Guard**: Meta's open-weights safety classifier.
- **CleanLab**: For detecting label noise in safety datasets.

---

## 6. Key References

1.  **Meta AI (2024)**: *Llama 3 Safety technical report*.
2.  **Inan et al. (2023)**: *Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations*.
3.  **Microsoft (2023)**: *Presidio — Data Protection and De-identification SDK* (open-source, github.com/microsoft/presidio).
