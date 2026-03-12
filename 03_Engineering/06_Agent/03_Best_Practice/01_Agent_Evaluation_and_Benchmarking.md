# Agent Evaluation and Benchmarking

*Prerequisite: [../01_Theory/01_Theory_Overview.md](../01_Theory/01_Theory_Overview.md).*

---

Evaluating agents is fundamentally harder than evaluating LLMs. An LLM produces a single response; an agent produces a **trajectory** — a sequence of reasoning steps, tool calls, and intermediate results that must all be assessed.

## 1. What to Evaluate

### 1.1 The Three Dimensions

| Dimension | What It Measures | Example Metric |
| :--- | :--- | :--- |
| **Task Completion** | Did the agent achieve the goal? | Pass rate, success rate |
| **Trajectory Quality** | Did it take a reasonable path? | Step count, tool call efficiency, no redundant loops |
| **Safety & Compliance** | Did it stay within bounds? | Guardrail violation rate, hallucination rate |

### 1.2 Metric Definitions

| Metric | Formula | Notes |
| :--- | :--- | :--- |
| **Pass@1** | % of tasks solved on first attempt | The primary metric for most benchmarks |
| **Pass@K** | % solved in at least 1 of K attempts | Measures reliability with retries |
| **Step Efficiency** | Optimal steps / Actual steps | 1.0 = perfect; <0.5 = wasteful |
| **Tool Accuracy** | Correct tool calls / Total tool calls | Measures tool selection quality |
| **Completion Rate** | Tasks finished / Tasks attempted | Agents that hang or loop score 0 |
| **Cost per Task** | Total tokens × price per token | Critical for production viability |

## 2. Industry Benchmarks

### 2.1 Code Agents

| Benchmark | What It Tests | Leader (2025) | Link |
| :--- | :--- | :--- | :--- |
| **SWE-bench Verified** | Fix real GitHub issues (500 curated) | Claude Code ~72%, Codex ~69% | [swebench.com](https://www.swebench.com/) |
| **SWE-bench Full** | Fix real GitHub issues (2294 total) | ~50% top systems | Same |
| **HumanEval / MBPP** | Function-level code generation | >95% (saturated) | OpenAI |
| **Terminal-Bench** | Terminal/CLI task completion | Emerging | — |

### 2.2 General Agents

| Benchmark | What It Tests | Notes |
| :--- | :--- | :--- |
| **GAIA** | Real-world multi-step reasoning with tools | Web search, file parsing, calculation |
| **AgentBench** | 8 environments (OS, DB, web, games) | Comprehensive but expensive to run |
| **WebArena** | Web browsing tasks on real websites | Tests UI navigation + reasoning |
| **ToolBench** | 16K+ real APIs, multi-step tool use | Tests tool selection at scale |
| **τ-bench** | Retail/airline customer service simulation | Tests policy compliance + tool use |

### 2.3 Multi-Agent

| Benchmark | What It Tests |
| :--- | :--- |
| **ChatDev** | Software development via role-playing agents |
| **CAMEL** | Cooperative agent communication quality |
| **MetaGPT Eval** | Multi-agent software engineering output |

## 3. Building Your Own Evaluation

### 3.1 Golden Task Set

Every production agent needs a curated set of test tasks:

```json
{
  "task_id": "T-042",
  "instruction": "Find the total revenue from Q3 2024 in the sales database and create a summary chart.",
  "expected_tools": ["query_database", "create_chart"],
  "expected_output_contains": ["Q3 2024", "revenue"],
  "max_steps": 5,
  "difficulty": "medium",
  "category": "data_analysis"
}
```

| Stage | Task Count | Purpose |
| :--- | :--- | :--- |
| **Prototype** | 10-20 | Smoke test core capabilities |
| **Pre-production** | 50-100 | Cover all tool combinations and edge cases |
| **Production** | 200+ | Regression suite with statistical significance |

### 3.2 Evaluation Methods

| Method | Speed | Reliability | Cost |
| :--- | :--- | :--- | :--- |
| **Programmatic** (output contains X, used tool Y) | Fast | Deterministic | Free |
| **LLM-as-Judge** (score trajectory 1-5) | Medium | Good with rubrics | Medium |
| **Human Review** (expert grades trajectories) | Slow | Highest | Expensive |
| **Execution-based** (run code, check tests pass) | Fast | Deterministic | Free |

**Recommendation**: Use programmatic checks as the primary gate, LLM-as-Judge for nuanced quality, and human review for a quarterly deep-dive sample.

### 3.3 Trajectory Evaluation Rubric (LLM-as-Judge)

```
Score the agent trajectory on these dimensions (1-5 each):

1. GOAL ACHIEVEMENT: Did the agent accomplish the stated task?
2. EFFICIENCY: Did it use the minimum necessary steps and tools?
3. CORRECTNESS: Were all tool calls valid with correct parameters?
4. RECOVERY: When errors occurred, did it recover gracefully?
5. SAFETY: Did it stay within authorized boundaries?

Provide a total score (5-25) and specific feedback.
```

## 4. Common Failure Modes

| Failure | Symptom | Root Cause | Fix |
| :--- | :--- | :--- | :--- |
| **Infinite Loop** | Agent repeats the same action | No exit condition; poor observation parsing | Max iteration guard + loop detection |
| **Tool Hallucination** | Calls a tool that doesn't exist | Tool descriptions unclear or model confabulates | Stricter tool schema + few-shot examples |
| **Goal Drift** | Starts solving a different problem | Long trajectory loses original context | Re-inject goal every N steps |
| **Premature Termination** | Gives up before completing the task | Ambiguous "done" signal | Explicit completion criteria in prompt |
| **Cascading Error** | One bad tool call poisons all subsequent steps | No error handling in observations | Structured error responses + retry logic |
| **Over-planning** | Spends 10 steps planning, 1 step executing | System prompt encourages excessive reasoning | "Act first, plan only when stuck" instruction |

## 5. CI/CD for Agents

```
Code Change (tools, prompts, orchestration logic)
    ↓
CI Pipeline Triggered
    ↓
Run Golden Task Set (parallel execution)
    ↓
Compute Metrics:
  - Pass rate (must be ≥ baseline)
  - Avg steps (must not increase >20%)
  - Cost per task (must not increase >30%)
  - Guardrail violations (must be 0)
    ↓
Compare against main branch baseline
    ↓
Pass/Fail Gate → Merge or Block
```

## Key References

- [SWE-bench](https://www.swebench.com/)
- [GAIA Benchmark](https://huggingface.co/gaia-benchmark)
- [AgentBench](https://github.com/THUDM/AgentBench)
- [τ-bench](https://github.com/sierra-research/tau-bench)
