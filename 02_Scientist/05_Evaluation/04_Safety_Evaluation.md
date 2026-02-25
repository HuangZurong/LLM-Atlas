# Safety Evaluation: Red-Teaming & Bias

*Prerequisite: [01_Benchmarks_Taxonomy.md](01_Benchmarks_Taxonomy.md). For alignment context, see [../04_Post_Training/02_Alignment/02_Preference_Alignment/07_Safety_Fine_Tuning.md](../04_Post_Training/02_Alignment/02_Preference_Alignment/07_Safety_Fine_Tuning.md).*

---

## 1. Factuality & Hallucination

### 1.1 TruthfulQA
- **What**: 817 questions designed to elicit common misconceptions (e.g., "What happens if you swallow gum?").
- **Metric**: % of answers that are both truthful AND informative.
- **Key finding**: Larger models are often *less* truthful — they learn to reproduce popular misconceptions more fluently.

### 1.2 SimpleQA (OpenAI, 2024)
- **What**: Short factual questions with unambiguous, verifiable answers.
- **Metric**: Correct / Incorrect / "I don't know" (measures calibration).

---

## 2. Social Bias & Fairness

| Benchmark | What it Tests | Format |
| :--- | :--- | :--- |
| **BBQ** | Ambiguity-based bias across 9 social categories (age, gender, race, etc.) | MCQ with ambiguous/disambiguated contexts |
| **BOLD** | Open-ended generation bias across 5 domains | Prompt completion + toxicity scoring |
| **WinoBias** | Gender bias in coreference resolution | Binary choice |

---

## 3. Red-Teaming & Adversarial Robustness

### 3.1 Manual Red-Teaming
Trained human adversaries attempt to elicit harmful outputs. Meta's Llama 3 report used domain experts across:
- Cybersecurity, CBRN (Chemical/Biological/Radiological/Nuclear), and child safety.

### 3.2 Automated Red-Teaming
- **GCG (Greedy Coordinate Gradient)**: Optimizes adversarial suffixes via gradient-based token substitution to bypass safety training.
- **AutoDAN**: Uses a **genetic algorithm** (crossover + mutation on prompt tokens) to evolve jailbreak prompts — it does NOT use an LLM as the attacker.
- **PAIR (Prompt Automatic Iterative Refinement)**: An **attacker LLM** iteratively refines jailbreak prompts against a target through multi-turn conversation.

---

## 4. Refusal Calibration

A critical balance: the model must refuse harmful requests but NOT over-refuse benign ones.
- **Over-refusal**: "Tell me how to make a cake" → "I cannot help with that" (false positive).
- **Under-refusal**: Complying with genuinely harmful requests (false negative).
- **Metric**: Plot Refusal Rate vs. Harm Category to find the optimal threshold.

---

## 5. Key References

1.  **Lin et al. (2021)**: *TruthfulQA: Measuring How Models Mimic Human Falsehoods*.
2.  **Parrish et al. (2021)**: *BBQ: A Hand-Built Bias Benchmark for Question Answering*.
3.  **Zou et al. (2023)**: *Universal and Transferable Adversarial Attacks on Aligned Language Models* (GCG).
4.  **Liu et al. (2023)**: *AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models*.
5.  **Chao et al. (2023)**: *Jailbreaking Black Box Large Language Models in Twenty Queries* (PAIR).
6.  **Meta AI (2024)**: *Llama 3 Safety Report*.
