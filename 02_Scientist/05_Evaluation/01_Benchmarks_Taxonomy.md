# Benchmarks Taxonomy: A Scientist's Map

*Prerequisite: None (Entry point for Evaluation module).*
*See Also: [../../04_Solutions/04_Evaluation_Loop.md](../../04_Solutions/04_Evaluation_Loop.md) (domain evaluation design), [../../03_Engineering/08_LLMOps/02_Practical/01_Automated_Evaluation_Runner.py](../../03_Engineering/08_LLMOps/02_Practical/01_Automated_Evaluation_Runner.py) (CI evaluation pipeline).*

---

## 1. General Reasoning & World Knowledge

| Benchmark | What it Tests | Format | SOTA Reference |
| :--- | :--- | :--- | :--- |
| **MMLU** | 57-subject knowledge (STEM, humanities, social sciences) | 4-choice MCQ | GPT-4: ~86% |
| **MMLU-Pro** | Harder MMLU with 10 choices + reasoning-heavy questions | 10-choice MCQ | Reduces noise from guessing |
| **GPQA** | Graduate-level science QA (physics, chemistry, biology) | 4-choice MCQ | Diamond set: expert-validated |
| **ARC-Challenge** | Grade-school science (hard subset) | 4-choice MCQ | — |
| **BBH (BIG-Bench Hard)** | 23 challenging tasks from BIG-Bench requiring multi-step reasoning | Free-form / MCQ | CoT essential |
| **HellaSwag** | Commonsense sentence completion | 4-choice MCQ | Near-saturated (>95%) |
| **WinoGrande** | Pronoun coreference resolution (commonsense) | Binary choice | Near-saturated |

---

## 2. Mathematical Reasoning

| Benchmark | What it Tests | Format | Notes |
| :--- | :--- | :--- | :--- |
| **GSM8K** | Grade-school math word problems | Free-form answer | 8.5K problems; CoT essential |
| **MATH** | Competition-level math (AMC/AIME difficulty) | Free-form with LaTeX | 5 difficulty levels |
| **Olympiad Bench** | International Math Olympiad level | Proof-based | Frontier-only |

> **Note on Minerva**: Minerva is a Google **model** (PaLM fine-tuned on math/science data), not a benchmark. The evaluation datasets it used include MATH, GSM8K, and custom STEM problem sets.

---

## 3. Code Generation & Software Engineering

| Benchmark | What it Tests | Format | Notes |
| :--- | :--- | :--- | :--- |
| **HumanEval** | Function-level Python generation | pass@k | 164 problems; OpenAI |
| **MBPP** | Basic Python programming | pass@k | 974 problems; Google |
| **SWE-bench** | Real GitHub issue resolution | Full repo patch | Tests agentic coding |
| **LiveCodeBench** | Contamination-free competitive programming | Continuously updated | Post-cutoff problems |

---

## 4. Long Context

| Benchmark | What it Tests | Format | Notes |
| :--- | :--- | :--- | :--- |
| **Needle-in-a-Haystack** | Single fact retrieval in long context | Exact match | Tests attention across 128K+ |
| **RULER** | Multi-hop reasoning over long documents | Various | More rigorous than NIAH |
| **LongBench** | Diverse long-context tasks (summarization, QA, code) | Task-specific | Chinese + English |

---

## 5. Instruction Following

| Benchmark | What it Tests | Format | Notes |
| :--- | :--- | :--- | :--- |
| **IFEval** | Strict format constraints ("write exactly 3 paragraphs") | Rule-based verification | No LLM judge needed |
| **MT-Bench** | Multi-turn conversation quality | LLM-as-Judge (GPT-4) | 8 categories |
| **AlpacaEval 2.0** | Single-turn instruction quality | Length-Controlled Win Rate | Corrects verbosity bias |

---

## 6. Multilingual

| Benchmark | What it Tests | Format | Notes |
| :--- | :--- | :--- | :--- |
| **C-Eval** | Chinese knowledge (52 subjects) | MCQ | The "Chinese MMLU" |
| **CMMLU** | Chinese multitask understanding | MCQ | Broader than C-Eval |
| **MGSM** | Multilingual grade-school math | Free-form | GSM8K in 10 languages |

---

## 7. Encoder / Retrieval Benchmarks

| Benchmark | What it Tests | Format | Notes |
| :--- | :--- | :--- | :--- |
| **GLUE / SuperGLUE** | NLU classification, entailment, similarity | Task-specific | Classic; mostly saturated |
| **BEIR** | Zero-shot information retrieval | nDCG@10 | 18 diverse IR datasets |
| **MTEB** | Massive Text Embedding Benchmark | Multiple metrics | The standard for embeddings |

---

## 8. Key References

1.  **Hendrycks et al. (2021)**: *Measuring Massive Multitask Language Understanding* (MMLU).
2.  **Cobbe et al. (2021)**: *Training Verifiers to Solve Math Word Problems* (GSM8K).
3.  **Chen et al. (2021)**: *Evaluating Large Language Models Trained on Code* (HumanEval).
4.  **Jimenez et al. (2024)**: *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*
5.  **Chiang et al. (2024)**: *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference*.
6.  **Rein et al. (2023)**: *GPQA: A Graduate-Level Google-Proof Q&A Benchmark*.
7.  **Suzgun et al. (2022)**: *Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them* (BBH).
