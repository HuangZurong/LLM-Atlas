# Automated Prompt Evaluation & CI/CD

*Prerequisite: [../01_Theory/02_Programmatic_Prompting.md](../01_Theory/02_Programmatic_Prompting.md).*

---

In industrial environments, prompts are not static. They change as models upgrade, requirements evolve, or edge cases are discovered.

A **Prompt CI/CD Pipeline** ensures that any change to a prompt doesn't degrade performance.

## 1. The Prompt Lifecycle

1. **Development**: Use **DSPy** or manual iteration to create a candidate prompt.
2. **Local Eval**: Run the prompt against a small "Golden Dataset" (10-50 cases).
3. **Pull Request**: Submit prompt changes to Git.
4. **CI Pipeline (Automated)**:
   - Run **Prompt Unit Tests** (e.g., check for JSON validity).
   - Run **LLM-as-a-Judge** on the full test suite (100+ cases).
   - Compare metrics (Accuracy, Cost, Latency) against the `main` branch.
5. **Approval**: Human reviewer looks at the "Diff" in quality.
6. **Deployment**: Update the `prompts.yaml` in the production environment.

## 2. CI Metric: Regression Testing

When you "fix" a prompt for one edge case, you often break three others.
- **Goal**: Maintain a **Golden Dataset** that represents the diversity of real user queries.
- **Fail Condition**: If the new prompt's score is >2% lower than the baseline, block the PR.

## 3. A/B Testing Prompts in Shadow Mode

Don't just deploy a new prompt. Run it in **Shadow Mode**:
1. Live traffic hits the `Production Prompt`.
2. A copy of the traffic hits the `Candidate Prompt`.
3. Compare outputs using a Judge model.
4. If `Candidate` wins >55% of the time, promote it.

## 4. Prompt Versioning Strategy

| Level | Method | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Simple** | Hardcoded Strings | Fast to start | Impossible to version/test |
| **Intermediate** | `prompts.yaml` in Git | Versioned, easy to audit | Requires code deployment |
| **Advanced** | Prompt Registry (e.g., Langfuse, Arize) | Instant updates, dynamic A/B | External dependency |

## 5. Summary Check-list for Industrial Prompts

- [ ] Does it have a **Version** tag?
- [ ] Has it been tested on **multiple temperatures**?
- [ ] Is it **model-agnostic**, or model-specific?
- [ ] Does it have an associated **Metric** and **Golden Set**?
- [ ] Does it handle **Prompt Injection** gracefully?
