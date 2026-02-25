# Security and Compliance Checklist for LLM Apps

*Prerequisite: [../01_Theory/01_LLM_Security.md](../01_Theory/01_LLM_Security.md).*

---

Securing an LLM application is an ongoing process. Use this checklist to audit your application across all layers of the stack.

## 1. Input Security (Prompt Hardening)

- [ ] **Delimiters**: Are user inputs wrapped in clear delimiters (e.g., `###`)?
- [ ] **Input Sanitization**: Are dangerous characters or sequences (e.g., `<script>`) stripped?
- [ ] **Injection Detection**: Is there a middleware/guardrail to block common injection patterns?
- [ ] **Few-shot Isolation**: Are few-shot examples separated from the user input?
- [ ] **Length Limits**: Is there a hard cap on input length to prevent context-overflow attacks?
- [ ] **Multi-turn Guard**: Does the system monitor for attacks that span multiple chat turns?

## 2. Model Configuration

- [ ] **System Prompt**: Does the system prompt explicitly forbid following instructions in user data?
- [ ] **Parameter Hardening**: Is `temperature` low (e.g., < 0.3) for critical tasks to reduce unpredictability?
- [ ] **Model Versioning**: Are you using specific model versions (not just `latest`) to ensure stable behavior?
- [ ] **Stop Sequences**: Are appropriate stop sequences configured to prevent the model from generating beyond its scope?

## 3. Output Security (Response Guardrails)

- [ ] **PII Redaction**: Is there a post-generation scan for leaked emails, credit cards, or SSNs?
- [ ] **Toxicity Filtering**: Are responses checked for hate speech, bias, or harmful content?
- [ ] **Format Validation**: Are structured outputs (JSON/XML) validated against a schema before use?
- [ ] **Hallucination Check**: For RAG systems, is there a faithfulness check against retrieved documents?
- [ ] **Fact Verification**: Are critical factual claims cross-referenced with a ground-truth database?

## 4. Agent and Tool Safety

- [ ] **Least Privilege**: Does each agent have the minimum set of tools required?
- [ ] **Sandboxed Tools**: Are code-execution or file-writing tools run in isolated containers (e.g., gVisor)?
- [ ] **Human-in-the-loop (HITL)**: Do destructive actions (delete, send, pay) require explicit human approval?
- [ ] **Parameter Sanitization**: Are tool parameters validated and sanitized (e.g., prevent SQL injection)?
- [ ] **Timeout Guards**: Are there hard timeouts on tool execution to prevent resource exhaustion?

## 5. Data Privacy & Compliance

- [ ] **Zero-Retention**: If using 3rd party APIs, are you opted into zero-data-retention for training?
- [ ] **Data Sovereignty**: Is data processed and stored in the legally required geographic region (e.g., EU)?
- [ ] **Data Encryption**: Is all sensitive data encrypted at rest (AES-256) and in transit (TLS 1.3)?
- [ ] **De-identification**: Is PII removed or masked from training/fine-tuning datasets?
- [ ] **Consent**: Are users explicitly informed when their data is processed by an LLM?

## 6. Observability & Audit

- [ ] **Immutable Logs**: Are all prompts, responses, and tool calls logged with a persistent ID?
- [ ] **Security Alerts**: Are there alerts for blocked injections or guardrail violations?
- [ ] **Audit Trail**: Is there a complete history of which user accessed which data via which model?
- [ ] **Traceability**: Can you trace a specific model output back to the specific retrieved documents (in RAG)?

## 7. Infrastructure Security

- [ ] **VPC Isolation**: Is the inference engine deployed inside a private network?
- [ ] **API Authentication**: Is every API call authenticated via JWT, OAuth2, or API keys?
- [ ] **Rate Limiting**: Are there per-user and per-organization limits on requests and tokens?
- [ ] **Secrets Management**: Are API keys stored in a secure vault (e.g., AWS Secrets Manager, Vault)?

## 8. Red Teaming & Testing

- [ ] **Adversarial Testing**: Have you performed manual red teaming to find "jailbreaks"?
- [ ] **Automated Scanning**: Do you regularly run security scanners like **Garak** or **PyRIT**?
- [ ] **Regression Testing**: Is security testing integrated into the CI/CD pipeline?
- [ ] **Incident Response**: Is there a documented plan for handling a data leak or model compromise?
