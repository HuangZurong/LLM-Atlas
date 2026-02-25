# Agent Cost and Latency Management

*Prerequisite: [../01_Theory/03_Workflow_Patterns.md](../01_Theory/03_Workflow_Patterns.md).*

---

Agents are the most expensive LLM pattern in production. A single agent task can involve 5-20 LLM calls, each with growing context windows. Without active management, costs and latency spiral out of control.

## 1. The Cost Model

### 1.1 Why Agents Are Expensive

```
Standard LLM Call:    1 request  × ~2K tokens  = ~2K tokens total
ReAct Agent (5 steps): 5 requests × ~4K tokens  = ~20K tokens total
                       (context grows each step)

Multi-Agent (3 workers + orchestrator):
  Orchestrator plan:     1 × 2K  =  2K
  Worker 1:              1 × 3K  =  3K
  Worker 2:              1 × 3K  =  3K
  Worker 3:              1 × 3K  =  3K
  Synthesis:             1 × 8K  =  8K
  ─────────────────────────────────────
  Total:                           19K tokens (minimum)
```

**The compounding problem**: In a ReAct loop, every step appends the full history (previous thoughts + tool calls + observations) to the next request. By step 5, you're sending 5x the context of step 1.

### 1.2 Cost Estimation Formula

```
Cost per agent task ≈ Σ (input_tokens_i × input_price + output_tokens_i × output_price)
                      for i in 1..N steps

Where input_tokens_i ≈ system_prompt + history_up_to_i + current_observation
```

### 1.3 Reference Costs (2025 Pricing)

| Model | Input (per 1M) | Output (per 1M) | 10-step Agent Task (est.) |
| :--- | :--- | :--- | :--- |
| GPT-4o | $2.50 | $10.00 | $0.15 - $0.50 |
| GPT-4o-mini | $0.15 | $0.60 | $0.01 - $0.03 |
| Claude Sonnet 4 | $3.00 | $15.00 | $0.20 - $0.60 |
| Claude Haiku 3.5 | $0.80 | $4.00 | $0.05 - $0.15 |

**Takeaway**: A 10-step agent on GPT-4o costs 10-50x more than a single LLM call. At 1000 requests/day, that's $150-$500/day on a single agent.

## 2. Cost Optimization Strategies

### 2.1 Tiered Model Strategy (The #1 Lever)

Use expensive models only where they matter:

| Role | Model Tier | Rationale |
| :--- | :--- | :--- |
| **Orchestrator / Planner** | Expensive (GPT-4o, Claude Sonnet) | Planning quality determines everything |
| **Workers / Executors** | Cheap (GPT-4o-mini, Haiku) | Narrow tasks with clear instructions |
| **Evaluator / Judge** | Medium (GPT-4o-mini with rubric) | Structured scoring doesn't need top-tier |
| **Router / Classifier** | Cheapest or embedding-based | Simple classification task |

**Impact**: 60-80% cost reduction with <5% quality loss in most cases.

### 2.2 Context Window Management

| Strategy | How | Savings |
| :--- | :--- | :--- |
| **Sliding Window** | Only keep last N steps in context, summarize older ones | 40-60% token reduction |
| **Observation Truncation** | Limit tool output to first 500 tokens; summarize if longer | 20-40% |
| **System Prompt Caching** | Use provider prefix caching (Anthropic, OpenAI) for static system prompts | 50-90% on cached portion |
| **Selective History** | Only include tool calls that are relevant to the current step | 30-50% |

### 2.3 Early Termination

```python
# Don't let agents run forever
BUDGET_GUARDS = {
    "max_iterations": 10,          # Hard stop
    "max_tokens": 50_000,          # Cumulative token budget
    "max_cost_usd": 0.50,          # Dollar cap per task
    "idle_detection": 3,           # Stop if same tool called 3x in a row
}
```

### 2.4 Caching

| Cache Type | What It Caches | Hit Rate |
| :--- | :--- | :--- |
| **Exact Match** | Identical tool calls with same args | Low (10-20%) |
| **Semantic Cache** | Similar queries → cached agent results | Medium (20-40%) |
| **Tool Result Cache** | Deterministic tool outputs (DB queries, API calls) | High (40-70%) |

**Rule**: Cache tool results aggressively. If `get_weather("NYC")` was called 5 minutes ago, don't call it again.

## 3. Latency Optimization

### 3.1 Latency Breakdown

```
Typical 5-step ReAct Agent:

Step 1: LLM call (800ms) + Tool call (200ms) = 1000ms
Step 2: LLM call (900ms) + Tool call (150ms) = 1050ms  (context grew)
Step 3: LLM call (1000ms) + Tool call (300ms) = 1300ms
Step 4: LLM call (1100ms) + Tool call (100ms) = 1200ms
Step 5: LLM call (1200ms) + no tool (final)   = 1200ms
─────────────────────────────────────────────────────────
Total:                                           5750ms
```

**Bottleneck**: LLM inference dominates. Each step is slower than the last because context grows.

### 3.2 Optimization Techniques

| Technique | Impact | Complexity |
| :--- | :--- | :--- |
| **Streaming** (first token) | Perceived latency drops 50%+ | Low |
| **Parallel Tool Execution** | If LLM requests 3 tools, run all concurrently | Medium |
| **Speculative Execution** | Pre-fetch likely tool results while LLM is thinking | High |
| **Smaller Models for Workers** | 2-3x faster inference | Low |
| **Context Compression** | Shorter context = faster inference | Medium |
| **Prompt Caching** | Cached prefix skips prefill computation | Low |

### 3.3 Parallel Tool Execution

Many LLMs (OpenAI, Anthropic) support requesting multiple tool calls in a single response. Execute them concurrently:

```python
# Sequential (bad): 3 tools × 200ms = 600ms
for call in tool_calls:
    result = await execute(call)

# Parallel (good): 3 tools × 200ms = 200ms (wall clock)
results = await asyncio.gather(*[execute(c) for c in tool_calls])
```

## 4. Monitoring Dashboard

### 4.1 Key Metrics

| Metric | Target | Alert |
| :--- | :--- | :--- |
| **Avg Steps per Task** | <6 | >10 |
| **Avg Cost per Task** | <$0.10 | >$0.50 |
| **p50 Latency** | <5s | >10s |
| **p95 Latency** | <15s | >30s |
| **Token Efficiency** | >0.5 (useful output tokens / total tokens) | <0.3 |
| **Budget Exhaustion Rate** | <5% of tasks hit the cap | >10% |
| **Cache Hit Rate** | >30% | <10% |

### 4.2 Cost Attribution

Track cost per dimension to identify optimization targets:

```
Total Agent Cost Breakdown:
├── Orchestrator/Planner:  35%  ← Use caching, reduce re-planning
├── Worker Execution:      40%  ← Use cheaper models
├── Evaluation/Judge:      15%  ← Use programmatic checks where possible
└── Synthesis:             10%  ← Acceptable
```

## 5. Decision Framework: When NOT to Use Agents

Agents are powerful but not always the right choice. Use simpler patterns when possible:

| If... | Use... | Not... |
| :--- | :--- | :--- |
| Task is a single LLM call | Direct prompt | Agent |
| Steps are fixed and predictable | Prompt Chaining | Agent |
| No tools needed | Standard LLM | Agent |
| Latency budget <2s | Pre-computed or cached response | Agent |
| Cost budget <$0.01/request | Single cheap model call | Agent |

**The golden rule**: An agent should only be used when the task genuinely requires dynamic planning and tool use. If you can hardcode the workflow, do it — it's cheaper, faster, and more reliable.
