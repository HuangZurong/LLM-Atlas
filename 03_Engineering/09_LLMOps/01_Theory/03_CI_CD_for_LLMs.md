# CI/CD for LLM Applications

*Prerequisite: [02_Observability_Evaluation.md](02_Observability_Evaluation.md).*

---

Continuous Integration and Continuous Deployment (CI/CD) for LLMs differ from traditional software due to the non-deterministic nature of model outputs. You need a pipeline that validates not just code, but also **prompts, retrieval quality, and model behavior**.

## 1. The LLM CI/CD Pipeline

```
Code/Prompt Change ──► Unit Tests ──► Eval Suite ──► Shadow Deployment ──► Canary Rollout ──► Production
```

### 1.1 Automated Evaluation (CI)
Every pull request that modifies a prompt, model parameter, or RAG strategy should trigger an automated evaluation.

- **Deterministic Tests**: Check for JSON schema compliance, regex patterns, or keyword presence.
- **Model-Based Eval**: Use an "LLM-as-a-Judge" to score the new version against a Golden Dataset.
- **Regression Detection**: Compare the new scores against the main branch baseline. If the "Faithfulness" score drops by >5%, block the merge.

### 1.2 Deployment Strategies (CD)
- **Blue-Green Deployment**: Switch 100% of traffic from the old version (Blue) to the new version (Green) after testing.
- **Canary Release**: Gradually shift traffic (e.g., 1% → 10% → 50% → 100%) to monitor for real-world regressions.
- **Shadow Mode (Mirroring)**: Run the new version in parallel with the current one. The new version processes real traffic, but its output is logged and compared, not shown to users.

## 2. Prompt Management (PromptOps)

Prompts should be treated as versioned artifacts, decoupled from the core application logic.

| Practice | Description | Benefit |
| :--- | :--- | :--- |
| **Prompt Registry** | A central store for all prompts (e.g., a Git repo or a DB) | Versioning and rollback capability |
| **Templating** | Using Jinja2 or similar for dynamic prompts | Cleaner code, easier testing |
| **Prompt Metadata** | Storing model, temperature, and version with each prompt | Full reproducibility |
| **Decoupled Release** | Updating a prompt via a configuration change, not a code deploy | Faster iteration cycles |

## 3. Tooling and Infrastructure

- **Version Control**: Git for code and prompts.
- **CI Runner**: GitHub Actions, GitLab CI, or Jenkins.
- **Eval Frameworks**: RAGAS, DeepEval, or Promptfoo.
- **Model Gateways**: Portkey, LiteLLM, or Martian for model-agnostic routing and failover.

**Key Rule**: Never deploy a prompt change without running it through the evaluation suite. Natural language is code that you don't fully control.
