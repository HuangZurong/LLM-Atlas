# Reasoning Model Strategies: o1 & DeepSeek-R1 (Part 3)

*Prerequisite: [02_Programmatic_Prompting.md](02_Programmatic_Prompting.md).*

---

The emergence of **Reasoning Models** (OpenAI o1, DeepSeek-R1) changes the fundamental rules of prompt engineering. These models have "Chain-of-Thought" baked into their inference process via Reinforcement Learning.

## 1. The "Less is More" Rule

With standard models (GPT-4o), we use CoT prompting ("Think step by step"). With Reasoning Models:
- **DO NOT** prompt for CoT. They do it automatically in a hidden "Reasoning" state.
- **DO NOT** provide too many examples. Over-prompting can restrict the model's "internal search space," leading to worse results.

## 2. Engineering for Reasoning Models

### 2.1 Delimiters and Constraints
Since these models "reason" for longer, they are better at following complex, multi-layered constraints.
- Use precise **XML tags** or **Markdown headers**.
- Be extremely explicit about the **Output Format**.

### 2.2 The "Reasoning Token" Budget
Engineering for o1 involves managing the `max_completion_tokens`.
- You need to allow enough tokens for the **internal reasoning** (which is billed but hidden) plus the final answer.
- **Rule of Thumb**: Set `max_completion_tokens` significantly higher than your expected output size.

## 3. Comparison Matrix: Standard vs. Reasoning

| Strategy | Standard Models (GPT-4o/Claude) | Reasoning Models (o1/R1) |
| :--- | :--- | :--- |
| **CoT** | Required ("Think step by step") | Counter-productive (Built-in) |
| **Few-Shot** | Highly effective for style/format | Less necessary for logic; use for style only |
| **Instructions** | Detailed, step-by-step guidance | High-level goal + strict constraints |
| **Latency** | Fast initial response | "Pensiveness" (Seconds/Minutes of thought) |

## 4. When to use Reasoning Models?
- **High Complexity**: Math, Code, Logic puzzles.
- **Zero-shot Hard Tasks**: Where Few-shot on standard models still fails.
- **NOT for**: Summarization, simple Q&A, or tasks where low latency is critical.
