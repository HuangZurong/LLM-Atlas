# Structured Output & Function Calling

*Prerequisite: [01_Foundations_and_Anatomy.md](01_Foundations_and_Anatomy.md).*

---

Structured Output transforms LLMs from "text generators" into "programmable interfaces". It is the foundation of Tool Use, Agent systems, and reliable data extraction.

## 1. The Spectrum of Output Control

```
Free Text → Markdown/XML → JSON Mode → JSON Schema → Function Calling → Tool Use (Agent)
  ↑                                                                              ↑
  Least constrained                                              Most constrained
```

| Method | Provider Support | Guarantee Level |
| :--- | :--- | :--- |
| **Prompt-based** ("Reply in JSON") | All | Best-effort. May break. |
| **JSON Mode** | OpenAI, Gemini | Valid JSON, no schema enforcement. |
| **JSON Schema (Structured Outputs)** | OpenAI, Gemini | Schema-conformant. Constrained decoding. |
| **Function Calling / Tool Use** | OpenAI, Anthropic, Gemini, Mistral | Structured function call; runtime executes. |

## 2. How Constrained Decoding Works

1. JSON Schema → compiled into a **Context-Free Grammar (CFG)**.
2. At each token step, logits are **masked** — invalid tokens set to `-inf`.
3. Model can only produce tokens leading to valid completion.

**Trade-off**: ~5-15% latency increase, reduced diversity. Almost always worth it for structured tasks.

## 3. Function Calling Architecture

```
User Message → LLM (with tool defs) → tool_call(name, args)
    → Runtime executes function → Result as tool_result message
    → LLM generates final response
```

### Tool Definition Anatomy

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current weather. Use when user asks about weather.",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
      },
      "required": ["city"]
    }
  }
}
```

**Critical**: The `description` is the "prompt" telling the model WHEN to call. Poor descriptions → wrong tool selection.

### Call Patterns

| Pattern | Example |
| :--- | :--- |
| **Single** | "What's the weather in Beijing?" |
| **Parallel** | "Compare weather in Beijing and Shanghai." |
| **Sequential** | "Find cheapest flight, then book it." |
| **Nested** | "Search DB; if no result, search web." |

## 4. Provider Comparison

| Aspect | OpenAI | Anthropic | Gemini |
| :--- | :--- | :--- | :--- |
| **Schema enforcement** | `response_format` (strict) | Best-effort | `response_schema` |
| **Tool choice** | `"auto"/"required"/{"name":"X"}` | `"auto"/"any"/{"name":"X"}` | `tool_config` |
| **Parallel calls** | Yes | Yes | Yes |

## 5. Production Patterns

### Graceful Degradation

```
Structured Output (constrained) → JSON Mode + validation → Free text + extraction → Error
```

### Common Failures

| Failure | Fix |
| :--- | :--- |
| **Refusal** | Check `refusal` field before parsing. |
| **Hallucinated enum** | Use strict/constrained mode. |
| **Truncation** | Increase `max_tokens`; simplify schema. |
| **Wrong tool** | Improve descriptions; add negative examples. |

## Key References

1. **OpenAI (2024)**: *Structured Outputs Guide*.
2. **Anthropic (2024)**: *Tool Use Documentation*.
3. **Willard & Louf (2023)**: *Efficient Guided Generation for LLMs*.