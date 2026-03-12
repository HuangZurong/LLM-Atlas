# Context Quality & Evaluation

*Prerequisite: [01_Context_Architecture_Patterns.md](01_Context_Architecture_Patterns.md).*

---

Context quality determines whether the model has the right information, in the right amount, in the right order. Poor context quality is the most common root cause of LLM application failures.

## 1. The Four Dimensions of Context Quality

| Dimension | Definition | Failure Mode |
| :--- | :--- | :--- |
| **Relevance** | Injected content is pertinent to the current query | Irrelevant RAG chunks distract the model |
| **Completeness** | All information needed to answer is present | Model hallucinates missing facts |
| **Conciseness** | No redundant or low-value content | Token waste, attention dilution |
| **Ordering** | Most important content at optimal positions | Lost in the Middle failures |

These dimensions are in tension: maximizing completeness conflicts with conciseness. Context engineering is the art of finding the right balance.

---

## 2. Evaluation Methods

### 2.1 Needle-in-a-Haystack (NIAH)

Insert a specific fact ("needle") at a known position in a long document ("haystack"). Ask the model to retrieve it.

**Setup**:
```python
def niah_test(model_fn, context_length: int, needle_depth: float) -> bool:
    """
    needle_depth: 0.0 = start, 0.5 = middle, 1.0 = end
    Returns True if model correctly retrieves the needle.
    """
    needle = "The secret project codename is AURORA."
    haystack = generate_filler_text(context_length)
    insert_pos = int(len(haystack) * needle_depth)
    context = haystack[:insert_pos] + needle + haystack[insert_pos:]
    response = model_fn(context, "What is the secret project codename?")
    return "AURORA" in response
```

**Interpretation**:
- Score > 95% across all depths → model is reliable at this context length
- Score drops at 40–60% depth → Lost in the Middle; use Sandwich Pattern
- Score drops beyond X tokens → X is your effective context limit

### 2.2 Context Relevance Score (LLM-as-Judge)

Use a cheap model to score whether retrieved context is relevant to the query.

**Prompt template**:
```
You are evaluating whether retrieved context is relevant to a user query.

Query: {query}

Retrieved Context:
{context}

Score the relevance from 0 to 1:
- 1.0: Directly answers or strongly supports answering the query
- 0.5: Partially relevant, contains some useful information
- 0.0: Irrelevant, does not help answer the query

Output only a JSON object: {"score": <float>, "reason": "<one sentence>"}
```

**Threshold**: Discard chunks with relevance score < 0.3 before injection.

### 2.3 Context Utilization Rate

Measures what fraction of injected context the model actually references in its response.

**Approach**: After generation, use an attribution model (or LLM judge) to identify which parts of the context were used.

```
Utilization Rate = (tokens from context cited in response) / (total injected context tokens)
```

**Benchmark**: A well-tuned RAG system should have utilization > 40%. Below 20% suggests over-injection.

### 2.4 Compression Quality (ROUGE-L)

When evaluating compression strategies, measure information preservation:

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
scores = scorer.score(original_text, compressed_text)
rouge_l = scores['rougeL'].fmeasure
# Target: ROUGE-L > 0.6 for acceptable compression quality
```

---

## 3. Context Debugging Checklist

When the model gives a wrong or incomplete answer, diagnose systematically:

1. **Is the answer in the context at all?**
   - Search the assembled context for the expected answer.
   - If absent → retrieval problem (RAG/memory), not a context engineering problem.

2. **Is the answer in the middle of a long context?**
   - Check the position of the relevant chunk.
   - If yes → apply Sandwich Pattern or move chunk to end.

3. **Is the context over-full?**
   - Check input token count vs. window size.
   - If >85% full → increase compression aggressiveness.

4. **Is there conflicting information?**
   - Search for contradictory statements in the context.
   - If yes → deduplicate or add explicit conflict resolution instructions.

5. **Is the system prompt being followed?**
   - Test with a minimal context (system prompt + query only).
   - If the model follows instructions with minimal context but not full context → context is overriding the system prompt.

6. **Is compression losing critical information?**
   - Compare model output with compressed vs. uncompressed context.
   - If uncompressed gives correct answer → compression quality is too low.

---

## 4. A/B Testing Context Strategies

To validate context changes, run controlled experiments:

**Metrics to track**:
| Metric | How to Measure |
| :--- | :--- |
| Answer correctness | LLM-as-Judge or human eval on test set |
| Context relevance score | Automated (see §2.2) |
| Input token count | API response metadata |
| Cost per query | Token count × price |
| Latency (TTFT) | API timing |
| User satisfaction | Thumbs up/down, session length |

**Minimum sample size**: 200–500 queries per variant for statistical significance on correctness metrics.

**Rollout strategy**: Shadow mode first (log both variants, serve only control), then 10% → 50% → 100%.

---

## Key References

1. **Es, S., et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation.** *arXiv:2309.15217*.
2. **Hsieh, C., et al. (2024). RULER: What's the Real Context Size of Your Long-Context Language Models?** *arXiv:2404.06654*.
3. **Gao, Y., et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey.** *arXiv:2312.10997*.
