# Prompt Engineering: The Industrial Standard (Part 1)

*Prerequisite: [../../01_LLMs/01_Theory/03_API_Mechanics.md](../../01_LLMs/01_Theory/03_API_Mechanics.md).*
*See Also: [04_Structured_Output_and_Function_Calling.md](04_Structured_Output_and_Function_Calling.md) (JSON/Tool Use patterns), [../../../04_Solutions/03_Domain_Data_Strategy.md](../../../04_Solutions/03_Domain_Data_Strategy.md) (prompt templates for domain data).*

---

Prompt engineering has evolved from a "black art" into a rigorous engineering discipline. This guide covers the anatomy and classic patterns required for production-grade LLM applications.

## 1. Prompt Anatomy & Architecture

In professional development, a prompt is not a single string but a **structured message sequence**.

### 1.1 The Components of an Industrial Prompt
| Component | Purpose | Industrial Insight |
| :--- | :--- | :--- |
| **Persona (Role)** | Sets the behavioral baseline. | Don't just say "You are an expert". Specify years of experience and specific domain (e.g., "Senior SRE specializing in Kubernetes"). |
| **Context** | Bound the model's knowledge. | Use XML-style tags (`<context>...</context>`) to help the model distinguish background from instructions. |
| **Instruction** | The core command. | Use strong, imperative verbs. Place instructions at the **end** of long contexts to avoid "Lost in the Middle". |
| **Few-Shot Examples** | Calibrates format and style. | **Quality > Quantity**. 3 diverse examples with reasoning steps (CoT) are better than 10 repetitive ones. |
| **Guardrails** | Explicit negative constraints. | "DO NOT mention competitors", "Output ONLY valid JSON". |
| **Output Schema** | Enforces structure. | Always provide a schema (Pydantic or JSON Schema) even if using `json_mode`. |

### 1.2 The Sandwich Pattern (Context Handling)
Long-context models (128k+) still suffer from attention dilution.
```markdown
[Instruction] Analyze the following logs for security anomalies.
[Context]      <10,000 lines of logs...>
[Reminder]     Focus on SSH brute force patterns. Output as a summary table.
```

---

## 2. Core Reasoning Patterns

### 2.1 Chain-of-Thought (CoT)
Forcing the model to "think" before answering.
- **Zero-shot CoT**: "Let's think step by step."
- **Structured CoT**: "First, analyze the input. Second, verify constraints. Third, generate the response."

### 2.2 Self-Consistency & Majority Voting
Sampling the same prompt multiple times (T > 0) and taking the most frequent answer. Essential for high-stakes math/logic tasks.

### 2.3 ReAct (Reasoning + Acting)
The backbone of Agents. The model generates a **Thought**, executes an **Action** (Tool call), observes the **Observation**, and repeats.

---

## 3. Cost & Performance Optimization

### 3.1 Prompt Caching (Prefix Caching)
Modern providers (OpenAI, Anthropic, DeepSeek) charge significantly less for **Cached Prompts**.
- **Strategy**: Keep the stable part of your prompt (System Prompt, Examples) at the beginning. Move dynamic data (User Query) to the end.
- **Impact**: Up to 90% reduction in latency and 50% reduction in cost for repeated prefix hits.

### 3.2 Prompt Compression
Using LLMs to "summarize" prompts or removing filler words to fit within narrow context windows or reduce per-token billing.
