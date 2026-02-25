# LLMOps Production Checklist

*Prerequisite: [../01_Theory/01_Maintenance.md](../01_Theory/01_Maintenance.md).*

---

This checklist covers the operational requirements for running LLM applications in production.

## 1. Evaluation & Quality Control

- [ ] **Golden Dataset**: At least 50 high-quality (input, ground-truth) pairs covering all major features.
- [ ] **Automated CI Eval**: Every PR triggers an evaluation suite with LLM-as-a-Judge.
- [ ] **Regression Detection**: New versions are blocked if scores drop below the main branch baseline.
- [ ] **JSON Schema Validation**: All structured outputs are validated against a Pydantic/JSON schema.
- [ ] **Manual Review Loop**: A process for human experts to review and label "difficult" or "failed" cases.

## 2. Observability & Tracing

- [ ] **Distributed Tracing**: Full request flow (retrieval, reasoning, tool calls) is logged to a tool like LangSmith or Langfuse.
- [ ] **Token Usage Tracking**: Tokens per request, per user, and per feature are monitored for billing.
- [ ] **Latency Monitoring**: TTFT (Time to First Token) and TPOT (Time Per Output Token) are tracked.
- [ ] **Error Classification**: Specific monitoring for rate limits (429), timeouts, and model errors.
- [ ] **User Feedback**: Thumbs up/down or feedback text collection integrated into the UI.

## 3. Prompt & Model Management

- [ ] **Prompt Registry**: Prompts are stored and versioned outside of the application code.
- [ ] **Model Versioning**: Using pinned versions (e.g., `gpt-4o-2024-08-06`) instead of `latest`.
- [ ] **A/B Testing Framework**: Capability to run multiple prompt/model versions in parallel for live testing.
- [ ] **Fallback Strategy**: Automatic failover to a secondary model or static response if the primary model is down.

## 4. Cost & Performance

- [ ] **Token Limits**: Hard caps on input and output tokens to prevent runaway costs.
- [ ] **Prompt Caching**: Enabled for static prefixes (system prompts) to reduce cost and latency.
- [ ] **Quantization**: Self-hosted models are quantized (INT8/FP8) for optimal VRAM usage.
- [ ] **Rate Limiting**: Tiered rate limits per user/organization to prevent abuse.

## 5. Security & Safety

- [ ] **Prompt Injection Guard**: Middleware to detect and block malicious injection patterns.
- [ ] **PII Redaction**: Automatic masking of sensitive data (emails, credit cards) in logs and API calls.
- [ ] **Audit Trail**: Immutable logs of all prompt/response pairs with user context for forensic analysis.
- [ ] **HITL for Destructive Actions**: Human approval required for tools with side effects (delete, send, pay).
