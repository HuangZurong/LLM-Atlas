# Rejection Sampling (Best-of-N)

*Prerequisite: [../02_Preference_Alignment/01_Overview.md](../02_Preference_Alignment/01_Overview.md).*

Rejection Sampling, also known as **Best-of-N sampling**, is a simple yet powerful technique to improve model performance during post-training. It acts as a bridge between Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL).

> **Context**: In the DeepSeek-R1 pipeline, Rejection Sampling is used after the initial RL phase to collect high-quality data for the subsequent "User-Friendly" SFT phase.

---

## 1. How It Works

The core idea is to generate multiple candidate responses for a single prompt and use a reward signal to select only the best one.

1. **Sampling**: For each prompt in the training set, generate $N$ independent responses (typically $N=8, 16, 64,$ or even $256$) using a policy model $\pi_{\theta}$.
2. **Scoring**: Use a **Reward Model (RM)** or a **Verifiable Reward function** (e.g., a code compiler or math checker) to score all $N$ responses.
3. **Selection**: Select the response with the highest score (or responses above a certain threshold).
4. **Fine-Tuning**: Create a new SFT dataset consisting of `(Prompt, Best_Response)` pairs and fine-tune the model on this "distilled" high-quality data.

## 2. Why Use Rejection Sampling?

| Advantage | Explanation |
|:--|:--|
| **Simplicity** | Unlike PPO/GRPO, it doesn't involve complex RL optimization or multiple models in memory. |
| **Data Quality** | It filters out the "noise" of the policy's average performance, focusing only on its "best" capabilities. |
| **Stability** | It is essentially SFT on high-quality data, which is much more stable than policy gradient methods. |
| **Exploration** | By generating $N$ samples, the model "explores" the solution space. If it finds a correct answer even once in 64 tries, that successful path can be reinforced. |

## 3. Rejection Sampling vs. RLHF

| Feature | Rejection Sampling | RLHF (PPO/GRPO) |
|:--|:--|:--|
| **Learning Paradigm** | Supervised (Offline) | Reinforcement (Online) |
| **Reward Signal** | Used to *select* data | Used to *calculate gradients* |
| **Computational Cost** | High during data prep (N samples), low during training | High during training (runtime RM + Policy) |
| **Solution Quality** | Limited by the model's best sample in N | Can theoretically surpass any single sample via optimization |

## 4. Rejection Sampling in Modern Pipelines

### 4.1 Llama 2/3 Style
Meta uses iterative rounds of Rejection Sampling. In each round:
1. Generate samples from the current best model.
2. Rank them using the latest Reward Model.
3. Train the next model version on the top-ranked samples.

### 4.2 DeepSeek-R1 Style
DeepSeek-R1 uses Rejection Sampling as a **data filter** for cold-start and final SFT:
1. The model performs RL to gain reasoning capabilities.
2. Rejection Sampling is performed on a large instruction pool to find responses that follow reasoning patterns *and* are user-friendly.
3. These "best-of-both-worlds" samples are used for a final SFT stage.

## 5. Mathematical Intuition (Simplified)

If a model has a 10% chance of solving a hard math problem, the probability that it fails to solve it in $N$ independent tries is:
$P(\text{fail all}) = (1 - 0.1)^N$

- For $N=1$: $90\%$ failure.
- For $N=64$: $0.9^{64} \approx 0.1\%$ failure.

By using $N=64$, we turn a 10% success rate into a **99.9% success rate** for the training data. This "over-sampling" allows the model to learn from its rare successes.

## 6. Key Trade-offs

1. **Diversity vs. Quality**: If you only take the single best response for every prompt, the model might become repetitive or lose the ability to generate diverse outputs.
2. **Reward Model Bias**: If the Reward Model has a bias (e.g., preferring long answers), Rejection Sampling will amplify this bias into the new SFT dataset.
3. **Inference Cost**: Generating $N=256$ responses for 100k prompts is extremely expensive (requires massive GPU hours).

## 7. Implementation Tips

- **Temperature Control**: Use a higher temperature (e.g., $T=0.7$ or $1.0$) during sampling to encourage diversity and exploration.
- **Deduplication**: If multiple samples reach the same high reward (e.g., both get "Correct" in math), keep the shorter or more efficient one.
- **Iterative Refinement**: The model trained on Round 1's best samples becomes the generator for Round 2, leading to a "self-improvement" loop (Expert Iteration).

---

## 8. Key References

1. **Bai et al. (2022)**: *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback* (Anthropic, Best-of-N).
2. **Touvron et al. (2023)**: *Llama 2: Open Foundation and Fine-Tuned Chat Models* (Rejection Sampling in Llama 2).
3. **Yuan et al. (2023)**: *Scaling Relationship on Learning Mathematical Reasoning with Large Language Models* (RFT).
