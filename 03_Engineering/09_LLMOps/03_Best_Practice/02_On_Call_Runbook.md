# LLM On-Call Runbook

This runbook provides step-by-step procedures for on-call engineers responding to LLM application incidents.

---

## 1. Severity Classification

| Severity | Definition | Response Time | Example |
| :--- | :--- | :--- | :--- |
| **P0 — Critical** | Service down or producing harmful output. | < 15 min | Model returning PII, total API outage. |
| **P1 — High** | Major degradation affecting most users. | < 30 min | Latency 5x baseline, 50%+ error rate. |
| **P2 — Medium** | Partial degradation, workaround exists. | < 2 hours | One retrieval source down, quality dip on edge cases. |
| **P3 — Low** | Minor issue, no user impact. | Next business day | Logging gap, non-critical metric anomaly. |

---

## 2. Incident Response Flowchart

```
Alert Fired
    ↓
1. Acknowledge alert (PagerDuty/Slack)
    ↓
2. Classify severity (P0-P3)
    ↓
3. Is it a safety issue (harmful output, PII leak)?
    ├── YES → Activate Circuit Breaker (kill switch) → Escalate to Security
    └── NO → Continue diagnosis
    ↓
4. Check provider status page (OpenAI/Anthropic/AWS)
    ├── Provider outage → Activate fallback model → Monitor
    └── Provider healthy → Internal issue
    ↓
5. Diagnose (see Section 3)
    ↓
6. Fix / Mitigate
    ↓
7. Verify recovery (run eval suite)
    ↓
8. Write post-mortem (P0/P1 only)
```

---

## 3. Diagnostic Playbooks by Symptom

### 3.1 High Latency (TTFT or TPOT above SLO)

```
Check order:
1. Provider API latency → Dashboard or `curl` timing test.
2. Retrieval latency → Vector DB query time in traces.
3. Prompt size → Check if context window is bloated (token count spike).
4. Batch queue depth → If using continuous batching, check queue backlog.

Common fixes:
- Enable/verify prompt caching.
- Reduce retrieved document count (top_k).
- Switch to smaller model for non-critical paths.
- Scale up inference replicas.
```

### 3.2 High Error Rate (4xx / 5xx)

```
Check order:
1. 429 (Rate Limit) → Check token/request quotas. Enable request queuing.
2. 400 (Bad Request) → Schema change? Check tool definitions and response_format.
3. 500 (Provider Error) → Provider-side. Activate fallback.
4. Timeout → Increase timeout; check if prompt is too long.

Common fixes:
- Implement exponential backoff with jitter.
- Activate secondary model provider.
- Check for recent deployment that changed prompt templates.
```

### 3.3 Quality Degradation (Eval Score Drop)

```
Check order:
1. Was there a model version change? (Provider silent update)
   → Pin model version (e.g., gpt-4o-2024-08-06).
2. Was there a prompt or retrieval change?
   → Diff prompt registry; check recent commits.
3. Has the query distribution shifted? (Drift)
   → Check embedding similarity metrics.
4. Has the knowledge base changed?
   → Check last index update; verify document integrity.

Common fixes:
- Rollback to previous prompt version.
- Re-run eval suite against last known good config.
- If drift: collect new examples, update Golden Dataset.
```

### 3.4 Safety Incident (Harmful / Leaked Output)

```
IMMEDIATE ACTIONS (within 5 minutes):
1. Activate kill switch — return static fallback response.
2. Preserve evidence — snapshot the trace (query, context, response).
3. Notify security team and management.

Investigation:
1. Was it prompt injection? → Check user input for injection patterns.
2. Was it training data leakage? → Check if output matches verbatim training data.
3. Was it guardrail bypass? → Review guardrail logs for that request.

Recovery:
1. Patch the specific vulnerability (add filter rule, update guardrail).
2. Run adversarial eval suite before re-enabling.
3. Mandatory post-mortem within 24 hours.
```

---

## 4. Kill Switch Protocol

Every production LLM app must have a kill switch — a mechanism to instantly disable LLM generation and return a safe fallback.

```python
# Pseudocode
if KILL_SWITCH_ENABLED:  # Feature flag (LaunchDarkly, env var, Redis key)
    return {
        "response": "Service is temporarily unavailable. Please try again later.",
        "source": "fallback",
        "trace_id": trace_id,
    }
```

**Kill switch triggers:**
- Any P0 safety incident.
- Error rate > 80% for > 5 minutes.
- Manual activation by on-call engineer.

---

## 5. Post-Mortem Template (P0/P1)

```markdown
## Incident: [Title]
- **Date**: YYYY-MM-DD
- **Duration**: X minutes
- **Severity**: P0/P1
- **Impact**: [Users affected, requests failed]

## Timeline
- HH:MM — Alert fired.
- HH:MM — Acknowledged by [engineer].
- HH:MM — Root cause identified.
- HH:MM — Fix deployed.
- HH:MM — Recovery confirmed.

## Root Cause
[One paragraph describing the actual root cause.]

## What Went Well
- [e.g., Kill switch worked as expected.]

## What Went Wrong
- [e.g., Alert was delayed by 10 minutes due to metric lag.]

## Action Items
- [ ] [Preventive action] — Owner: [name] — Due: [date]
```

---

## 6. Escalation Matrix

| Situation | Escalate To |
| :--- | :--- |
| Provider outage > 30 min | Engineering Manager + Provider support ticket |
| PII/safety leak confirmed | Security Lead + Legal (if user data involved) |
| Cost anomaly (>3x daily budget) | Engineering Manager + Finance |
| Quality drop with no obvious cause | ML/LLM team lead for deep investigation |