# Best Practice: Multi-Model Routing Strategies

*Prerequisite: [01_Architecture_Decision_Matrix.md](01_Architecture_Decision_Matrix.md).*

A "Router" is an intelligent layer that sits in front of your LLMs to choose the best model for a given request.

---

## 1. Why Route?

- **Optimized Performance**: Send math to DeepSeek-R1, creative writing to Claude, and classification to Llama-8B.
- **Failover / Redundancy**: If OpenAI is down, route to Anthropic automatically.
- **Dynamic Pricing**: Route to the cheapest provider that satisfies your quality constraints.

## 2. Types of Routers

### 2.1 Keyword/Regex Router (Heuristic)
If query contains "math", route to Model A. If "SQL", Model B.
- **Pros**: Zero latency, 100% predictable.
- **Cons**: Brittle, fails on ambiguous queries.

### 2.2 Semantic Router (Embedding-based)
Compute the embedding of the user query and compare it to "category centroids" in a vector space.
- **Pros**: Fast (~20ms), handles varied phrasing well.
- **Cons**: Requires maintained reference embeddings.

### 2.3 LLM-based Router (The "Dispatch" Model)
Use a very small, fast model (e.g., GPT-4o-mini) to categorize the intent of the user.
- **Pros**: Extremely accurate, can handle complex logic.
- **Cons**: Adds latency and token cost to every request.

## 3. Implementation Blueprint

```python
# Pseudo-code for a Semantic Router
from semantic_router import Route, RouteLayer

# 1. Define Routes
math_route = Route(name="math", utterances=["calculate...", "solve equation..."])
code_route = Route(name="code", utterances=["write python...", "fix bug..."])

# 2. Setup Layer
layer = RouteLayer(encoder=my_encoder, routes=[math_route, code_route])

# 3. Route Query
def process_query(query):
    decision = layer(query)
    if decision.name == "math":
        return call_deepseek_r1(query)
    elif decision.name == "code":
        return call_deepseek_coder(query)
    else:
        return call_general_model(query)
```

## 4. Advanced: The "MoE-at-Scale" Architecture
For large organizations, routing is often centralized:

```
        User Query
            ↓
    ┌───────────────┐
    │ Gateway/Proxy │ (Auth, Rate Limit)
    └───────────────┘
            ↓
    ┌───────────────┐
    │ Router Engine │ (Semantic + Cost Logic)
    └───────────────┘
     ↙      ↓      ↘
[GPT-4o] [Llama] [Local vLLM]
```

## 5. Metrics for Success
- **Routing Accuracy**: How often does the router pick the "best" model?
- **Routing Latency**: Overhead added by the router itself.
- **Cost Reduction %**: Total savings compared to a single-model baseline.
