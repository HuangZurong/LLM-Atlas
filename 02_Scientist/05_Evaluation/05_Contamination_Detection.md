# Contamination Detection: Ensuring Scientific Validity

*Prerequisite: [02_Evaluation_Methodology.md](02_Evaluation_Methodology.md).*

---

## 1. Why Contamination Matters

If a model has "seen" benchmark questions during training, its scores are meaningless. This is the single biggest threat to the credibility of LLM evaluation.

---

## 2. Detection Methods

### 2.1 N-Gram Overlap (Standard)
The most widely used method. Scan the training corpus for exact substring matches against benchmark datasets.
- **Window size**: Typically 13-gram (GPT-4 report) or 8-gram (Llama 3).
- **Threshold**: If a training document shares a contiguous N-gram with a benchmark question+answer, it is flagged.
- **Limitation**: Misses paraphrased contamination.

### 2.2 Membership Inference (MIN-K% PROB)
Shi et al. (2023) proposed **MIN-K% PROB**: instead of using raw loss, compute the average log-likelihood of the $k\%$ least likely tokens in a sequence.
$$\text{MIN-K\% PROB}(x) = \frac{1}{|S_k|} \sum_{t \in S_k} \log P_\theta(x_t | x_{<t})$$
Where $S_k$ is the set of tokens with the lowest $k\%$ predicted probabilities. **Contaminated examples have higher MIN-K% PROB** (even their "hard" tokens are well-predicted).
- **Advantage**: Works on black-box models (only needs token probabilities, not training data access).

### 2.3 Perturbation-Based Detection
Slightly modify benchmark questions (e.g., change numbers in a math problem) and check if performance drops sharply.
- **Logic**: A model that memorized the original will fail on perturbations; a model that learned the skill will generalize.
- **Example**: GSM8K question with "24 apples" → change to "37 apples". If accuracy drops >20%, suspect contamination.

---

## 3. Paraphrased (Indirect) Contamination

The hardest case. The model never saw the exact benchmark text, but saw:
- A blog post discussing the benchmark question.
- A textbook that contains the same problem with different wording.
- Synthetic data generated from the benchmark.

**Mitigation**: There is no perfect solution. Best practices include:
- Embedding similarity check (flag if cosine similarity > 0.9 with any benchmark item).
- Continuously refreshing benchmarks (LiveCodeBench, Chatbot Arena).

---

## 4. Industry Practices

| Organization | Method | Details | Notes |
| :--- | :--- | :--- | :--- |
| **OpenAI (GPT-4)** | Substring match | Reported per-benchmark contamination rates | GPT-4 Technical Report Appendix C |
| **Meta (Llama 3)** | Token overlap | 8-gram token-level matching | Flagged "dirty" and "clean" splits |
| **Google (PaLM 2)** | Exact + near-duplicate | Variable window | Used both exact and fuzzy matching |

> **Note**: The exact n-gram window sizes vary across reports and are not always explicitly stated. The GPT-4 report describes contamination analysis but does not specify "13-gram" as a fixed parameter. Always check the original paper for precise methodology.

---

## 5. Key References

1.  **OpenAI (2023)**: *GPT-4 Technical Report*, Appendix C (Contamination Analysis).
2.  **Shi et al. (2023)**: *Detecting Pretraining Data from Large Language Models*.
3.  **Oren et al. (2023)**: *Proving Test Set Contamination in Black Box Language Models*.
