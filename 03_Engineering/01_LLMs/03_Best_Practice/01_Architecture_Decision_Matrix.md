# Best Practice: Model Selection Decision Matrix

*Prerequisite: [../01_Theory/05_Engineering_Paradigm.md](../01_Theory/05_Engineering_Paradigm.md).*

Choosing the right model for a production system is a balance of performance, cost, and latency. Use this matrix to guide your decision.

---

## 1. The "Big Three" Tiers

| Tier | Use Case | Recommended Models |
|---|---|---|
| **Tier 1: Intelligence Leader** | Complex reasoning, high-stakes decisions, zero-shot complex tasks | GPT-4o, Claude 3.5 Sonnet, DeepSeek-V3 |
| **Tier 2: Efficiency Leader** | Routine chat, summarization, extraction, high-volume classification | GPT-4o-mini, Gemini 1.5 Flash, Llama-3-8B |
| **Tier 3: Reasoning Specialist** | Math, deep logical chains, coding, self-correction tasks | OpenAI o1, DeepSeek-R1 |

## 2. Decision Trees

### 2.1 Latency vs. Quality
- **Requirement: Instant UI response (<500ms TTFT)**
    - Use a Tier 2 model (mini/flash) or a local small model (7B-8B).
- **Requirement: High-quality reasoning (latency >2s acceptable)**
    - Use a Tier 1 or Tier 3 model.

### 2.2 Cost vs. Scale
- **Volume < 10k requests/day**
    - Use API providers (OpenAI/Anthropic). Infrastructure overhead of self-hosting is not worth it.
- **Volume > 100k requests/day**
    - Consider self-hosting open-weights models (DeepSeek, Llama) on specialized inference engines (vLLM) to significantly reduce unit cost.

### 2.3 Data Privacy
- **Highly Sensitive (Legal, Health, Internal Trade Secrets)**
    - Self-host on private VPC/On-premise.
- **General Public Data**
    - API providers are fine (check for zero-retention policies).

## 3. The "Waterfall" Strategy (Recommended)

Don't use your most expensive model for everything. Implement a waterfall:

1.  **Cache First**: Check if a similar query was answered recently.
2.  **Small Model First**: Try to solve the task with a Tier 2 model (e.g., GPT-4o-mini).
3.  **Validation**: Use a cheap heuristic or another small model to check the output quality.
4.  **Escalation**: If quality is insufficient, pass the query + small model output to a Tier 1 model for "refinement."

## 4. Benchmarking Your Specific Use Case

Generic benchmarks (MMLU) are often misleading. Perform a **Task-Specific Evaluation**:

1.  Collect 100 real user queries.
2.  Get "Golden Answers" from your best model (or humans).
3.  Run the 100 queries against 3 candidate models.
4.  Compare using **LLM-as-a-Judge** on a scale of 1-5.
5.  Plot: **Score vs. Cost** and **Score vs. Latency**.

Choose the model at the "elbow" of the curve.
