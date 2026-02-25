# LLM API Fundamentals

*Prerequisite: [02_Tokenization_and_Cost.md](02_Tokenization_and_Cost.md).*

Understanding how to interact with LLM APIs at a deep level is the foundation of high-performance engineering. This document covers sampling parameters, streaming, and structured generation.

---

## 1. The Sampling Mechanism

LLMs do not produce a single answer; they produce a probability distribution over the entire vocabulary for the next token. Sampling is how we pick from that distribution.

### 1.1 Temperature ($T$)
- **How it works**: Scales the logits before the softmax. $P_i = \frac{\exp(z_i/T)}{\sum \exp(z_j/T)}$.
- **High T (0.8 - 1.2)**: Flattens the distribution, making less likely tokens more probable. Result: Creative, diverse, but prone to hallucinations.
- **Low T (0.1 - 0.3)**: Sharpens the distribution, making the most likely token dominate. Result: Deterministic, factual, but repetitive.
- **T = 0**: Equivalent to "Greedy Search" — always picks the single highest probability token.

### 1.2 Top-p (Nucleus Sampling)
- **How it works**: Sums tokens in descending order of probability until the total probability hits $p$. Discards all other tokens.
- **Why use it**: Prevents the model from picking extremely low-probability tokens that might lead to gibberish, even at high temperature.
- **Recommendation**: Adjust either $T$ or Top-p, but rarely both at the same time.

### 1.3 Frequency & Presence Penalties
- **Presence Penalty**: Penalizes a token if it has appeared *at least once*. Encourages the model to talk about new topics.
- **Frequency Penalty**: Penalizes a token proportionally to *how many times* it has already appeared. Prevents verbatim repetition of phrases.

## 2. Streaming & Latency

### 2.1 Server-Sent Events (SSE)
Standard LLM APIs use SSE to stream tokens. This is crucial for **Perceived Latency**:
- **TTFT (Time to First Token)**: The delay until the first character appears.
- **TPOT (Time Per Output Token)**: The speed at which the model "types."

### 2.2 Token-level vs. Chunk-level
Models generate token by token, but APIs often group tokens into chunks to reduce HTTP overhead.

## 3. Structured Generation

### 3.1 JSON Mode
The API ensures the output is valid JSON.
- **Constraint**: You must mention "JSON" in the prompt for it to work.
- **Limitation**: It only ensures *syntactic* validity, not adherence to a specific schema.

### 3.2 Function Calling / Tool Use
The model outputs a structured call instead of a text response.
- **Logic**: Prompt → Model sees tool definitions → Model generates `{"name": "get_weather", "args": {"city": "Paris"}}` → Application executes → Result sent back to model.
- **Reliability**: Modern models (GPT-4o, Claude 3.5, DeepSeek-V3) are fine-tuned specifically for high-reliability tool calling.

### 3.3 Constrained Decoding (Logit Bias & Grammars)
For 100% reliability, tools like **Outlines** or **SGLang** use regex or CFGs (Context-Free Grammars) to mask the vocabulary at each step.
- **How it works**: The engine calculates which tokens are valid according to the schema *before* sampling. Invalid tokens are given a probability of zero.
- **Benefit**: Zero JSON syntax errors, ever.
- **Limitation**: Requires access to logit manipulation (supported by vLLM, not supported by most hosted APIs like OpenAI).

### 3.4 Partial JSON Streaming
When streaming a JSON response, the client receives fragments like `{"name": "A`.
- **Engineering Task**: Use a "Partial JSON Parser" to display data to the user as soon as a key-value pair is complete, rather than waiting for the closing `}`.

## 4. Best Practices for API Integration
- **Idempotency**: Use `seed` or `request_id` to get reproducible results (where supported).
- **Graceful Error Handling**: Distinguish between "Retryable" (500, 429) and "Non-Retryable" (400, 403) errors.
- **Prompt Caching**: Always structure prompts with static content (System Prompt, Examples) at the beginning to trigger KV cache reuse.


---

## Key References

1. **Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). The Curious Case of Neural Text Degeneration.** *International Conference on Learning Representations*.
2. **OpenAI (2023). OpenAI API Documentation: Completions.** *OpenAI Platform Documentation*.
3. **Anthropic (2024). Claude API Reference: Messages API.** *Anthropic Documentation*.
4. **Kwon, W., et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention.** *Proceedings of the 29th Symposium on Operating Systems Principles*, 611–626.
