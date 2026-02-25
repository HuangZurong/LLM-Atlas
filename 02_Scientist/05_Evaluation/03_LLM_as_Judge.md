# LLM-as-Judge: Scalable Evaluation

*Prerequisite: [02_Evaluation_Methodology.md](02_Evaluation_Methodology.md).*

---

## 1. Why LLM-as-Judge?

Human evaluation is the gold standard but doesn't scale. LLM-as-Judge uses a strong model (typically GPT-4) to evaluate weaker models, achieving ~80% agreement with human annotators.

---

## 2. Major Frameworks

### 2.1 MT-Bench
- **Format**: 80 multi-turn questions across 8 categories (Writing, Roleplay, Reasoning, Math, Coding, Extraction, STEM, Humanities).
- **Scoring**: GPT-4 rates each response 1-10 with a written explanation.
- **Multi-turn**: The second turn tests the model's ability to follow up, correct, or refine.

### 2.2 AlpacaEval 2.0
- **Format**: 805 single-turn instructions. A judge compares the candidate model's output against a reference (GPT-4 Turbo).
- **Key innovation — Length-Controlled Win Rate (LC)**: Raw win rate is biased toward longer responses. LC fits a generalized linear model (GLM) to deconfound output length from quality, producing a win rate that reflects quality independent of verbosity.
- **Judge version matters**: Results can shift significantly depending on the judge model version (GPT-4-0613 vs GPT-4-Turbo). Always report the exact judge model used.

### 2.3 Arena-Hard
- **Format**: 500 challenging prompts extracted from real Chatbot Arena conversations where top models disagreed.
- **Purpose**: High discriminative power — separates models that static benchmarks cannot distinguish.

---

## 3. Known Biases

| Bias | Description | Mitigation |
| :--- | :--- | :--- |
| **Position Bias** | Judge prefers the response shown first | Randomize order; evaluate both orderings |
| **Verbosity Bias** | Longer responses rated higher regardless of quality | Length-Controlled metrics (AlpacaEval 2.0) |
| **Self-Enhancement** | GPT-4 rates its own outputs higher | Use diverse judges; cross-model evaluation |
| **Style Bias** | Preference for markdown, bullet points, formal tone | Include style-diverse references |

---

## 4. Best Practices

1. **Swap positions**: Always evaluate `(A, B)` and `(B, A)` and average.
2. **CoT Judging**: Ask the judge to reason before scoring — reduces random errors.
3. **Multi-judge**: Use 2-3 different judge models and take majority vote.
4. **Calibration**: Include known-quality anchor responses to detect judge drift.

---

## 5. Key References

1.  **Zheng et al. (2023)**: *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*.
2.  **Dubois et al. (2024)**: *Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators*.
3.  **Li et al. (2024)**: *From Crowdsourced Data to High-Quality Benchmarks: Arena Hard and BenchBuilder Pipeline*.
