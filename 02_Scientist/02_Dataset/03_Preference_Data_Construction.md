# Preference Data Construction: The RLHF Foundation

*Prerequisite: [02_Instruction_Data_Engineering.md](02_Instruction_Data_Engineering.md). For alignment algorithms (RLHF, DPO), see [../04_Post_Training/02_Alignment/](../04_Post_Training/02_Alignment/).*

---

## 1. The Anatomy of Preference Data

Preference data is the signal used to align model behavior with human values (Helpfulness, Honesty, Harmlessness). Unlike SFT, which uses single (Instruction, Response) pairs, preference learning uses **comparisons**.

### 1.1 Pairwise Comparison Format
Standard format: `(Prompt, Response_Chosen, Response_Rejected)`
- **Chosen ($y_w$)**: The better response.
- **Rejected ($y_l$)**: The inferior response.

---

## 2. Preference Modeling: Bradley-Terry (BT)

Most Reward Models (RM) are based on the **Bradley-Terry model**, which assumes the probability of preferring $y_w$ over $y_l$ is:
$$P(y_w \succ y_l) = \sigma(r(x, y_w) - r(x, y_l))$$
Where $r(x, y)$ is the scalar reward score for a given prompt and response.

---

## 3. Sourcing Preference Data

### 3.1 Human Feedback (RLHF) — InstructGPT Case Study
OpenAI's InstructGPT (Ouyang et al., 2022) is the canonical reference. Their exact dataset sizes:
- **SFT data**: ~13,000 prompt-response pairs written by human labelers.
- **Reward Model data**: ~33,000 pairwise comparisons (labelers ranked 4-9 model outputs per prompt).
- **PPO rollout prompts**: ~31,000 prompts from the API (no human labels needed).

Key insight: The RM training set is **2.5x larger** than the SFT set — ranking is cheaper than writing.

### 3.2 AI Feedback (RLAIF) — Constitutional AI
Anthropic's **Constitutional AI (CAI)** (Bai et al., 2022) introduced a two-phase approach:
1. **SL-CAI (Supervised Learning)**: The model critiques its own harmful response using a set of principles ("the Constitution"), then revises it. The final revised response becomes SFT training data.
2. **RL-CAI**: A preference model (PM) is trained using AI-generated comparisons — the model evaluates its own outputs against the Constitution, replacing human labelers for harmlessness judgments. The PM then trains the policy via RL.

> **Citation**: Bai et al. (2022), *Constitutional AI: Harmlessness from AI Feedback*, Anthropic.

**UltraFeedback Protocol** (Cui et al., 2023): Scores responses across 4 dimensions — **Instruction Following, Truthfulness, Honesty, and Helpfulness** — using GPT-4 as judge, then constructs preference pairs from the highest vs. lowest scoring responses.
- **CoT Prompting for Judging**: Asking the AI judge to explain *why* one response is better before assigning a score.

---

## 4. DPO vs. Reward Model Data

### 4.1 Reward Model Data (PPO)
Requires a massive diversity of prompts to train a robust reward predictor that can "generalize" to unseen responses.

### 4.2 DPO Data (Direct Preference Optimization)
DPO doesn't require a separate reward model. It uses the same pairwise comparisons to directly optimize the policy.
- **Critical Requirement**: The `Rejected` response must be "plausible but incorrect" or "lower quality," not just gibberish. If the delta between $y_w$ and $y_l$ is too large, the model doesn't learn fine-grained preferences.

---

## 5. Modern Data Curation: Argilla & Open-Source
- **Argilla**: A standard tool for managing preference labeling workflows.
- **Datasets**: UltraFeedback, HH-RLHF (Anthropic), Nectar.

---

## 6. Key References

1.  **Ouyang et al. (2022)**: *Training language models to follow instructions with human feedback* (InstructGPT).
2.  **Rafailov et al. (2023)**: *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*.
3.  **Cui et al. (2023)**: *UltraFeedback: Boosting Language Models with High-quality Feedback*.
