# Best Practice: LLM Cost Optimization

*Prerequisite: [../01_Theory/02_Tokenization_and_Cost.md](../01_Theory/02_Tokenization_and_Cost.md).*

In production, LLM costs can spiral out of control. Use these patterns to keep them manageable.

---

## 1. Input Token Reduction

Input tokens are the #1 driver of cost in RAG and Agent systems.

- **Selective Retrieval**: Don't inject 20 documents if 5 are enough. Use a Reranker to select only top-K high-relevance chunks.
- **Context Compression**: Use tools like LLMLingua to remove redundant filler words from retrieved context before sending to the model.
- **System Prompt Caching**: Place your long system prompt and few-shot examples at the *beginning* of the message array to take advantage of provider-side prefix caching.

## 2. Output Token Management

Output tokens are 3x–10x more expensive than input tokens.

- **Constrain Output**: "Summarize in 50 words" vs "Summarize."
- **Stop Sequences**: Set stop sequences (e.g., `\n`, `}`) to prevent the model from rambling.
- **Model Choice**: Reasoning models (o1/R1) generate many "hidden" tokens. Use them only when deep thinking is required.

## 3. Engineering Patterns for Savings

### 3.1 Prompt Chaining over Monolithic Prompts
Instead of one massive prompt that does 5 things, use 5 small prompts.
- **Benefit**: You only pay for the "expensive" model on the one sub-task that needs it.

### 3.2 Semantic Caching
Use a vector database to store (Prompt Embedding, Response).
- If a new query is semantically similar (e.g., >0.95 similarity), return the cached response.
- **Result**: Zero model cost for common questions.

### 3.3 Batching
Use the **OpenAI Batch API** or equivalent for non-real-time tasks (data labeling, translation, batch summarization).
- **Benefit**: 50% discount on standard token pricing.

## 4. Monitoring & Governance

- **Rate Limiting by User**: Prevent a single user/client from draining your entire token budget.
- **Cost Allocation Tags**: Track usage per project, team, or feature.
- **Alerting**: Set daily/monthly spend thresholds.

## 5. Decision Matrix: When to Fine-Tune for Cost
Fine-tuning a small model (8B) to replace a large model (GPT-4o) makes sense when:
1. Your prompt is massive (>5k tokens) but the task is repetitive.
2. Your volume is high (>1M requests/month).
3. The small model can reach 95%+ of the large model's accuracy after fine-tuning.
