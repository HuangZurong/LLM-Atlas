# Iterative Post-Training (Self-Improvement)

*Prerequisite: [../02_Preference_Alignment/02_PPO.md](../02_Preference_Alignment/02_PPO.md), [../02_Preference_Alignment/03_DPO.md](../02_Preference_Alignment/03_DPO.md).*

---

Modern LLM alignment is no longer a one-pass process (SFT → RLHF). Instead, it has evolved into an **iterative cycle** where the model acts as its own generator and judge to recursively improve its performance.

---

## 1. The Core Loop

The basic structure of iterative post-training consists of alternating between **Data Generation** and **Model Optimization**:

1. **Generation**: The current model $\pi_t$ generates multiple responses for a prompt set.
2. **Filtering/Labeling**: Responses are evaluated using a Reward Model, AI Judge, or Ground Truth.
3. **Training**: The model is updated using the filtered data (via SFT, DPO, or RL) to create $\pi_{t+1}$.
4. **Repeat**: $\pi_{t+1}$ becomes the new generator for the next round.

## 2. Key Frameworks

### 2.1 Expert Iteration (ExIt)
Originally from AlphaGo, applied to LLMs (e.g., in STaR).
- **Process**: Generate → Verify → SFT on correct answers.
- **Key Idea**: If the model finds a correct reasoning path for a difficult problem, learning from that "lucky" success makes it more likely to succeed in the future.

### 2.2 Self-Reward / Self-Alignment
The model acts as its own Judge (e.g., **Self-Rewarding Language Models** by Meta).
- **Process**:
    1. Model generates responses.
    2. Model scores its own responses using a rubric.
    3. Model is updated (e.g., via DPO) using its own preferences.
- **Risk**: Reward hacking or model collapse if the model's judgment is flawed.

### 2.3 Iterative DPO (ReST / SPIN)
- **ReST (Reinforced Self-Training)**: Google's approach using iterative filtering and DPO.
- **SPIN (Self-Play Fine-Tuning)**: The model "plays" against its previous version. It tries to generate responses that a judge cannot distinguish from ground-truth data, while the judge (also the model) tries to tell them apart.

## 3. Why Iterate?

| Reason | Explanation |
|:--|:--|
| **Distribution Shift** | Standard RLHF often suffers when the policy model drifts too far from the data the Reward Model was trained on. Iterating keeps the RM and Policy aligned. |
| **Bootstrapping** | A model can learn from its own "best" outputs that weren't present in the original human-labeled dataset. |
| **Efficiency** | Iterative DPO is often more stable and easier to scale than long-running PPO sessions. |
| **Reasoning** | For math and code, iteration is essential to explore diverse reasoning paths and "fix" mistakes. |

## 4. Case Study: DeepSeek-R1 Pipeline

DeepSeek-R1 is a masterclass in iterative post-training:

1. **R1-Zero**: Pure RL without SFT. It showed that reasoning can emerge through iteration.
2. **Cold Start**: A small, high-quality SFT to stabilize the model.
3. **Reasoning-Oriented RL**: Iteratively improving reasoning via GRPO.
4. **Rejection Sampling + SFT**: Using the RL model to generate 600k "perfect" samples, then fine-tuning on them.
5. **General RL**: A final round of RL for alignment and helpfulness.

## 5. Challenges in Iteration

### 5.1 Model Collapse / Inbreeding
If a model only learns from its own outputs, it may lose diversity, stop exploring new solutions, and eventually amplify its own errors.
- **Solution**: Inject a small percentage of high-quality human data in every round.

### 5.2 Reward Hacking
The model finds a way to get a high score from the Reward Model (e.g., by repeating key phrases) without actually solving the task.
- **Solution**: Update the Reward Model iteratively (Online RLHF) or use verifiable rewards (math/code).

### 5.3 Over-Optimization
Training too hard on a specific iteration can lead to "forgetting" general capabilities.
- **Solution**: Use KL-divergence constraints to keep the model close to its predecessor.

## 6. Implementation Strategy

To implement an iterative pipeline:
1. **Infrastructure**: You need a high-throughput inference engine (like vLLM) to generate millions of samples quickly between training rounds.
2. **Evaluation**: Robust automated benchmarks are required to decide when to move to the next iteration.
3. **Diversity**: Use diverse prompt sets (General, Math, Code, Safety) to prevent the model from becoming a "specialist" in only one area.

---

## 7. Key References

1. **Gulcehre et al. (2023)**: *Reinforced Self-Training (ReST) for Language Modeling*.
2. **Singh et al. (2024)**: *Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models* (ReST-EM).
3. **Pang et al. (2024)**: *Iterative Reasoning Preference Optimization* (Iterative DPO).
