# Evaluation Methodology: Rigor in Measurement

*Prerequisite: [01_Benchmarks_Taxonomy.md](01_Benchmarks_Taxonomy.md).*

---

## 1. Evaluation Paradigms

### 1.1 Prompting Strategies
| Strategy | Description | When to Use |
| :--- | :--- | :--- |
| **Zero-shot** | No examples in prompt | Tests raw generalization |
| **Few-shot (k-shot)** | k examples provided as context | Standard for MMLU (5-shot), GSM8K (8-shot) |
| **Chain-of-Thought (CoT)** | "Let's think step by step" | Essential for math/reasoning |
| **Zero-shot CoT** | CoT without examples | Surprisingly effective post-GPT-4 |

### 1.2 Code Evaluation: pass@k
For code generation, we sample $k$ solutions and check if **any** pass all unit tests:
$$\text{pass@k} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$
Where $n$ = total samples, $c$ = correct samples. Standard: report pass@1 (greedy) and pass@10.

---

## 2. Statistical Rigor

### 2.1 The Variance Problem
A single MMLU run can vary by ±1-2% due to prompt formatting, random seed, and sampling temperature. Reporting a single number is **not scientific**.

### 2.2 Best Practices
- **Bootstrap Confidence Intervals**: Resample results 1000x to compute 95% CI.
- **Multiple Seeds**: Run at least 3 seeds for generation tasks.
- **Temperature = 0**: For reproducible MCQ benchmarks (MMLU, ARC). Use temperature > 0 only for pass@k.

---

## 3. Elo Rating: Chatbot Arena

### 3.1 Two Rating Methods
Chatbot Arena actually uses **two** approaches:

**a) Online Elo** (simplified, for display):
$$P(A \text{ wins}) = \frac{1}{1 + 10^{(R_B - R_A)/400}}$$
After each vote: $R_A' = R_A + K \cdot (S_A - E_A)$
Where $K$ = update factor, $S_A$ = actual outcome (1/0/0.5), $E_A$ = expected outcome.

**b) Bradley-Terry MLE** (official leaderboard):
Fits all pairwise comparisons simultaneously via maximum likelihood estimation. This is more statistically robust and is the method used for the published rankings.

> **Note**: "Elo" is a proper noun (named after Arpad Elo), not an acronym — it should not be written as "ELO."

### 3.2 Why Arena Matters
- **No contamination risk**: Fresh human prompts, not static benchmarks.
- **Holistic**: Tests the full user experience, not isolated capabilities.
- **Limitation**: Expensive, slow to converge for new models (~10K votes needed).

### 3.3 Prompt Template Sensitivity
Benchmark scores can vary significantly (1-5%) based on prompt template formatting alone. The same model evaluated with different prompt wrappers can rank differently on leaderboards. Always report the exact template used.

---

## 4. Benchmark Saturation & Design

### 4.1 The Saturation Problem
When SOTA approaches human performance (e.g., HellaSwag >95%), the benchmark loses discriminative power. Signs of saturation:
- Top-10 models within 1-2% of each other.
- Random prompt variation causes more variance than model differences.

### 4.2 Next-Generation Benchmark Design
- **Dynamic benchmarks**: LiveCodeBench, Chatbot Arena (continuously refreshed).
- **Process evaluation**: Reward correct reasoning steps, not just final answers.
- **Adversarial construction**: Human experts craft problems specifically to fool frontier models.

---

## 5. Key References

1.  **Chiang et al. (2024)**: *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference*.
2.  **Gao et al. (2023)**: *A Framework for Few-Shot Language Model Evaluation* (lm-evaluation-harness).
3.  **Jain et al. (2024)**: *LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code*.
