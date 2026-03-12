# LLMOps: Observability & Evaluation

*Prerequisite: [01_Maintenance.md](01_Maintenance.md).*

---

## 1. The LLMOps Lifecycle

```
    ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
    │ Development │ ───►  │ Evaluation  │ ───►  │ Deployment  │
    └─────────────┘       └─────────────┘       └─────────────┘
           ▲                                           │
           │              ┌─────────────┐              │
           └───────────── │ Monitoring  │ ◄────────────┘
                          └─────────────┘
```

The core challenge: **"Evaluation is the new Training."** In LLM applications, you spend more time designing the evaluation pipeline than training the model.

## 2. Evaluation Pipeline

### 2.1 Offline Evaluation (Pre-release)
Before deploying a new prompt or model version, run it against a static test set.

- **Golden Dataset**: A curated set of 50–200 representative (Input, Expected Output) pairs.
- **Regression Testing**: Ensuring the new version doesn't "break" previously working edge cases.
- **Cost/Latency Benchmarking**: Measuring the impact on infrastructure before going live.

### 2.2 LLM-as-Judge (The Modern Standard)
Using a high-capability model (GPT-4o or Claude 3.5 Sonnet) to evaluate the outputs of a smaller or newer model.

- **Criteria**: Score based on specific rubrics: _Accuracy (1-5), Helpfulness (1-5), Tone compliance_.
- **Pairwise comparison**: Show the judge two responses (A and B) and ask which is better.
- **Reference-based**: Give the judge the "ground truth" answer and ask if the model's response is semantically equivalent.

### 2.3 RAG-Specific Metrics (The RAG Triad)
If you are building a RAG system, you must measure three distinct components:

| Metric | What it measures | Framework |
|---|---|---|
| **Faithfulness** | Is the answer derived *only* from the retrieved context? (No hallucinations) | RAGAS / DeepEval |
| **Answer Relevance** | Does the answer actually address the user's query? | RAGAS |
| **Context Precision** | Are the retrieved documents actually relevant to the query? | RAGAS |

## 3. Observability & Tracing

Observability is about understanding *why* a model gave a specific response.

### 3.1 Tracing (The "X-Ray" for LLMs)
A single user query often triggers a complex chain of events:
`Query → Embed → Vector Search → Context Filtering → Prompt Synthesis → LLM Call → Tool Use → Final Response`.

**Tracing tools** (LangSmith, Langfuse, Arize Phoenix) log every step of this chain, allowing you to:
- Identify where latency is coming from.
- Debug where a hallucination entered the process.
- Inspect the exact prompt sent to the model after RAG injection.

### 3.2 Key Metrics to Track

| Category | Metrics |
|---|---|
| **User Experience** | TTFT (Time to First Token), TPS (Tokens per Second), E2E Latency |
| **Quality** | User thumbs up/down, Hallucination rate (via judge), Task completion rate |
| **Infrastructure** | Token usage per organization, GPU memory utilization, Cache hit rate |
| **Economics** | Cost per 1k requests, Token efficiency (output/input ratio) |

## 4. Prompt Management (PromptOps)

Prompts are code. They should be managed like code.

- **Prompt Registry**: A central database to store versioned prompts.
- **Decoupling**: Move prompts out of your application code (e.g., store them in a YAML file or a Prompt CMS) so they can be updated without a full code redeploy.
- **A/B Testing**: Run 90% of traffic on `v1` of a prompt and 10% on `v2` to compare performance in the real world.

## 5. The Continuous Improvement Loop

LLMOps is a flywheel:

1. **Log**: Record every user interaction in production.
2. **Curate**: Filter for "low satisfaction" or "high latency" cases.
3. **Analyze**: Use an LLM-as-judge or human labeler to find the root cause (Bad retrieval? Weak instruction? Model too small?).
4. **Fix**: Adjust the prompt, chunking strategy, or upgrade the model.
5. **Evaluate**: Run the fix against the Offline Eval set.
6. **Deploy**: Release the improved version.
