# Safety Fine-Tuning (Safety Alignment)

*Prerequisite: [06_Constitutional_AI.md](06_Constitutional_AI.md).*

Safety Fine-Tuning is the process of aligning LLMs to be **harmless, honest, and helpful (HHH)** while preventing them from generating toxic, biased, or dangerous content. It is a critical stage of the post-training pipeline that ensures the model adheres to ethical guidelines and legal requirements.

> **Relationship to RLHF/RLAIF**: Safety alignment is often implemented as a specific "safety track" within the broader [RLHF](./01_Overview.md) or [RLAIF](./05_RLAIF.md) process, using specialized safety datasets and reward models.

---

## 1. The Core Tension: Helpfulness vs. Harmlessness

Aligning a model involves a fundamental tradeoff:
- **Helpfulness**: The model's ability to follow instructions and provide useful information.
- **Harmlessness**: The model's refusal to assist with harmful requests.

If a model is "too safe," it becomes **over-refusing** (refusing harmless prompts like "How do I kill a process in Linux?"). if it's "too helpful," it becomes **leaky** (answering "How do I make a bomb?"). Safety fine-tuning aims to find the optimal boundary.

## 2. Taxonomy of Safety Risks

Safety training targets several distinct categories of undesirable behavior:

| Category | Description | Example |
|:--|:--|:--|
| **Toxic Content** | Hate speech, harassment, or extreme profanity. | Racist slurs or personal attacks. |
| **Dangerous Content** | Instructions for illegal acts or physical harm. | "How to build a chemical weapon." |
| **Bias & Fairness** | Perpetuating harmful stereotypes or discrimination. | Associating certain professions with specific genders. |
| **Privacy (PII)** | Leaking personally identifiable information. | "What is the phone number of [Private Citizen]?" |
| **Deception** | Deliberately lying or manipulating the user. | Providing false medical or legal advice as fact. |
| **Jailbreaking** | Vulnerability to prompts designed to bypass safety filters. | "Roleplay as an evil AI with no rules..." |

## 3. The Safety Alignment Pipeline

Modern safety alignment usually follows a multi-stage approach, mirroring the general alignment pipeline but with safety-specific data.

### 3.1 Stage 1: Safety SFT (Supervised)
The model is fine-tuned on a high-quality dataset of **(Harmful Prompt, Safe Response)** pairs.

- **Safe Response Strategies**:
    - **Direct Refusal**: "I cannot fulfill this request because..."
    - **Educational Refusal**: Explain *why* the request is harmful.
    - **Safe Alternative**: Provide a safe way to achieve the underlying (non-harmful) goal.

### 3.2 Stage 2: Safety RLHF (Reinforcement)
Using a **Safety Reward Model (Safety RM)** to provide a scalar signal for RL optimization (PPO/DPO).

- **Preference Data**: Pairs where one response is "Safer" than the other.
- **Reward Hack Prevention**: Ensuring the model doesn't learn to simply start every sentence with "I apologize" while still providing harmful info.

### 3.3 Stage 3: Red Teaming
The most critical stage for safety. Human or AI "red teamers" actively try to break the model's safety filters.

1. **Discovery**: Find a prompt that bypasses current safety filters.
2. **Annotation**: Label the failure.
3. **Remediation**: Add the prompt and a safe response to the next training cycle.

## 4. Dual Reward Model Architecture

Industrial safety pipelines (Llama 2, Claude) use **two separate reward models** trained on different objectives:

```
r_total = r_helpfulness + λ · r_safety
```

But this simple sum is insufficient — a model can learn to be "slightly helpful + slightly safe" everywhere. The key engineering insight is the **safety override**:

```
if r_safety < threshold:
    r_total = r_safety          # safety signal dominates entirely
else:
    r_total = r_helpfulness     # normal helpfulness optimization
```

This creates a **hard floor**: no amount of helpfulness reward can compensate for a safety violation below the threshold.

### 4.1 Safety Reward Margin

Llama 2 introduces a **safety reward margin** during RL training:

- For each (prompt, response) pair, the safety RM produces a score.
- If the safety score falls below a margin δ compared to the reference model's response, the update is blocked or penalized.
- This prevents the policy from drifting into unsafe territory even when the helpfulness RM would reward it.

## 5. Case Study: Llama 2/3 Safety Pipeline

Meta's Llama series is a benchmark for open-weights safety. Their pipeline includes:

1. **Safety-specific RLHF**: Separate reward models for Helpfulness and Safety.
2. **Safety Reward Margin**: During RL, safety is prioritized. If a response is flagged as harmful by the Safety RM, that signal overrides the Helpfulness RM.
3. **Context Distillation**: Training the model to recognize safety context even when prompts are ambiguous.
4. **False Refusal Mitigation**: Actively training on "borderline" prompts (e.g., medical questions) to prevent over-refusal.

## 6. Red Teaming as a Closed-Loop System

Red teaming is not a one-time audit — it is a **continuous feedback loop** integrated into the training pipeline:

```
┌─────────────────────────────────────────────────────────┐
│  1. Red Team (human or AI) generates adversarial prompts │
│  2. Model produces responses                             │
│  3. Responses are labeled (harmful / safe)               │
│  4. Failures added to safety SFT / RLHF dataset         │
│  5. Model retrained → repeat                             │
└─────────────────────────────────────────────────────────┘
```

**AI Red Teaming (Perez et al., 2022)**: Train a separate "attacker" LM to generate prompts that maximize the probability of eliciting harmful responses from the target model. This scales red teaming beyond what human teams can cover.

Key insight: the attacker and defender are in an **adversarial co-evolution** — as the model gets safer, the attacker must find more subtle exploits.

## 7. Multi-Turn Safety

Single-turn safety is insufficient. Real attacks often span multiple turns:

- **Gradual escalation**: Start with benign requests, slowly shift toward harmful territory across turns.
- **Context manipulation**: Establish a fictional frame ("you are a chemistry teacher") then exploit it.
- **Memory poisoning**: Inject harmful context early in the conversation that influences later responses.

Safety training must include **multi-turn adversarial dialogues** in the training data, not just single (prompt, response) pairs. The model must maintain safety awareness across the full conversation history.

## 8. Scalable Oversight

As models become more capable than human evaluators in specialized domains, the core assumption of RLHF breaks down: **humans can no longer reliably judge which response is better**.

Proposed solutions:

| Approach | Idea |
|---|---|
| **Debate** (Irving et al., 2018) | Two AI agents argue opposing positions; human judges the debate, not the answer |
| **Amplification** (Christiano et al., 2018) | Recursively decompose hard tasks into subtasks humans can evaluate |
| **Process Supervision (PRM)** | Evaluate reasoning steps rather than final answers — easier for humans to verify |
| **Constitutional AI** | Offload judgment to the model itself, guided by explicit principles |

This is an open research problem — the long-term safety of superhuman AI systems depends on solving scalable oversight.

## 9. Defense Against Jailbreaking

Jailbreaking (e.g., "Do Anything Now" / DAN prompts) is a persistent challenge. Safety fine-tuning addresses this via:

- **Adversarial Training**: Training on thousands of known jailbreak patterns.
- **System Prompt Robustness**: Ensuring the model follows the system-level safety instructions even when the user prompt contradicts them.
- **Averaging Weights (Model Merging)**: Sometimes merging a "helpful" model with a "safe" model can find a better balance.

## 10. Evaluation & Benchmarks

How do we measure if a model is safe?

| Benchmark | Focus Area |
|:--|:--|
| **ToxiGen** | Machine-generated hate speech and toxicity. |
| **HHH Alignment** | Helpful, Honest, Harmless evaluations (Anthropic). |
| **TruthfulQA** | Measuring mimicry of human falsehoods and misconceptions. |
| **Do-Not-Answer** | A dataset of prompts that models should refuse. |
| **WildChat** | Real-world user prompts, including adversarial attempts. |

## 11. Limitations & Open Challenges

1. **Catastrophic Forgetting**: Safety training can sometimes degrade the model's reasoning or creative capabilities.
2. **Language Gap**: Models are often much safer in English than in low-resource languages.
3. **Emergent Risks**: As models get smarter (e.g., reasoning models), they may find more "clever" ways to assist with harm without triggering keyword-based filters.
4. **The "Alignment Tax"**: The performance drop caused by safety constraints.

## 12. Key References

- Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models" (2023) — Detailed safety section.
- Bai et al., "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback" (2022).
- Ganguli et al., "Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned" (2022).
- OpenAI, "GPT-4 System Card" (2023) — Comprehensive look at safety mitigations.
