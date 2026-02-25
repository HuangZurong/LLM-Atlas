# LLM Alignment: Module Overview

*Prerequisite: [../01_FT/01_Theory/01_Introduction.md](../01_FT/01_Theory/01_Introduction.md) (fine-tuning basics).*
*See Also: [../../07_Paper_Tracking/04_Alignment_Frontiers.md](../../07_Paper_Tracking/04_Alignment_Frontiers.md) (latest alignment research), [../../../04_Solutions/06_Finetuning_Playbook.md](../../../04_Solutions/06_Finetuning_Playbook.md) (DPO/PPO business implementation).*

---

## 1. The 4-Stage Training Paradigm

Modern LLM development follows a four-stage pipeline, where each stage builds on the previous:

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Pre-training                                          │
│  Goal: Build world knowledge and language understanding         │
│  Data: Trillions of tokens from the web                         │
│  Output: A powerful but unaligned base model                    │
└─────────────────────────┬───────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: Instruction Fine-Tuning (SFT)                         │
│  Goal: Teach the model to follow instructions                   │
│  Data: High-quality (prompt, response) pairs                    │
│  Output: A model that can hold conversations                    │
│  Note: SFT quality sets the model's capability ceiling          │
└─────────────────────────┬───────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: Preference Alignment                                  │
│  Goal: Align behavior with human values and safety              │
│  Signal: Subjective human/AI preference comparisons             │
│  Methods: PPO, DPO, KTO, RLAIF, Constitutional AI              │
│  Question answered: "How should the model behave?"              │
└─────────────────────────┬───────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 4: Reasoning Alignment (RLVR)                            │
│  Goal: Develop deep reasoning and self-correction ability       │
│  Signal: Objective verifiable correctness (math, code)          │
│  Methods: GRPO + Verifier, Process Reward Models               │
│  Question answered: "How should the model think?"               │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Preference vs. Reasoning Alignment

| Dimension | Preference Alignment | Reasoning Alignment |
|---|---|---|
| Core Question | How should the model behave? | How should the model think? |
| Reward Source | Human/AI preference labels | Rule-based verifier |
| Signal Type | Subjective (better/worse) | Objective (correct/incorrect) |
| Applicable Tasks | Open-ended (style, safety, tone) | Closed-ended (math, code, logic) |
| Key Risk | Reward hacking on learned RM | Sparse reward signal |
| Representative Methods | PPO, DPO, KTO | GRPO + RLVR |
| Industrial Example | InstructGPT, Claude | DeepSeek-R1 |

## 3. Module Structure

### 01_Preference_Alignment/
Techniques for aligning model behavior with human values and preferences.

| File | Topic |
|---|---|
| [01_Overview.md](./02_Preference_Alignment/01_Overview.md) | RLHF pipeline, reward modeling, method comparison |
| [02_PPO.md](./02_Preference_Alignment/02_PPO.md) | Proximal Policy Optimization |
| [03_DPO.md](./02_Preference_Alignment/03_DPO.md) | Direct Preference Optimization |
| [04_KTO.md](./02_Preference_Alignment/04_KTO.md) | Kahneman-Tversky Optimization |
| [05_RLAIF.md](./02_Preference_Alignment/05_RLAIF.md) | Reinforcement Learning from AI Feedback |
| [06_Constitutional_AI.md](./02_Preference_Alignment/06_Constitutional_AI.md) | Rule-based self-critique |
| [07_Safety_Fine_Tuning.md](./02_Preference_Alignment/07_Safety_Fine_Tuning.md) | Safety-specific alignment |

### 02_Reasoning_Alignment/
Techniques for developing verifiable reasoning capabilities.

| File | Topic |
|---|---|
| [01_RLVR.md](./03_Reasoning_Alignment/01_RLVR.md) | Reinforcement Learning with Verifiable Rewards |
| [02_GRPO.md](./03_Reasoning_Alignment/02_GRPO.md) | Group Relative Policy Optimization |

### 03_Advanced_Topics/
Cross-cutting techniques applicable to both paradigms.

| File | Topic |
|---|---|
| [01_Rejection_Sampling.md](./04_Advanced_Topics/01_Rejection_Sampling.md) | Best-of-N sampling strategies |
| [02_Iterative_Training.md](./04_Advanced_Topics/02_Iterative_Training.md) | Online and iterative alignment |
| [03_Inference_Time_Compute.md](./04_Advanced_Topics/03_Inference_Time_Compute.md) | Scaling compute at inference |
| [04_Model_Merging.md](./04_Advanced_Topics/04_Model_Merging.md) | Merging aligned models |
