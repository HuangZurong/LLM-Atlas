# Scientist Operations (MLOps for LLMs)

*Prerequisite: [../../06_Deployment/01_Theory/02_Deployment_Architecture.md](../../06_Deployment/01_Theory/02_Deployment_Architecture.md).*
*See Also: [../03_Best_Practice/02_On_Call_Runbook.md](../03_Best_Practice/02_On_Call_Runbook.md) (incident handling), [../../../04_Solutions/10_Implementation_Roadmap.md](../../../04_Solutions/10_Implementation_Roadmap.md) (PoC to Production).*

---

Fine-tuning is not a one-time event. Maintaining a "live" LLM requires constant monitoring of distribution shifts and safety guardrails.

## 1. Monitoring Model Performance

- **Functional Metrics**: Tracking Latency, Throughput, and Token Utilization.
- **Behavioral Monitoring**:
  - **Response Toxicity**: Real-time checking for biased or harmful output.
  - **Embedding Distance**: Monitoring the semantic distance between inputs and a "safe/ideal" reference set.

## 2. Data & Concept Drift

- **Data Drift**: Shifts in the underlying input data distribution over time (e.g., new slang, emerging events).
- **Concept Drift**: Shifts in the relationship between input and expected output (e.g., changing user preferences).
- **Detection**: Using statistical tests on generated output distributions.

## 3. Retraining Strategies

- **Periodic Retraining**: Re-tuning the model weekly/monthly on cold-storage data.
- **Trigger-Based Retraining**: Initiated automatically when evaluation metrics fall below a specific threshold (e.g., accuracy drop in a specific category).

## 4. Safety Guardrails [2024 State-of-the-Art]

- **拦截器模型 (Interceptors)**: Deploying **Llama Guard 3** or **Shield Gemma** in front of the main LLM to filter inputs/outputs.
- **WILDGUARD**: Open-source toolkit for safety risks and jailbreak detection.

---

[Llama Guard: LLM-based Input-Output Safeguards](https://arxiv.org/abs/2312.06674)
