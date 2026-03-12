# LLM Security Incident Response Playbook

This playbook defines procedures for detecting, containing, and recovering from security incidents specific to LLM applications.

---

## 1. LLM-Specific Threat Categories

| Category | Description | Severity |
| :--- | :--- | :--- |
| **Prompt Injection** | Attacker manipulates model behavior via crafted input. | P0-P1 |
| **PII Leakage** | Model outputs personal data from training or context. | P0 |
| **Data Exfiltration** | Agent tool calls used to send data to external endpoints. | P0 |
| **Jailbreak** | User bypasses safety guardrails to produce harmful content. | P1 |
| **Model Denial of Service** | Crafted inputs cause extreme latency or resource exhaustion. | P1 |
| **Training Data Poisoning** | Malicious data injected into fine-tuning pipeline. | P1 (delayed impact) |

---

## 2. Incident Response Phases

### Phase 1: Detection (< 5 min)

**Automated Detection Signals:**
- Guardrail block rate spike (>3x baseline in 10 min window).
- PII regex match in model output logs.
- Anomalous tool call patterns (new endpoints, high frequency).
- User reports via feedback channel.

**Manual Detection:**
- Periodic red-team audit results.
- Security review of trace samples.

### Phase 2: Containment (< 15 min)

```
Severity Assessment
    ↓
Is PII or harmful content actively being served?
    ├── YES → IMMEDIATE: Activate kill switch
    │         Block the specific user/IP if targeted attack
    │         Preserve full trace evidence (DO NOT delete logs)
    └── NO → Enable enhanced logging (capture full I/O)
             Tighten guardrail thresholds temporarily
```

**Containment Actions by Threat:**

| Threat | Containment Action |
| :--- | :--- |
| **Prompt Injection** | Add input to blocklist; tighten input sanitization rules. |
| **PII Leakage** | Kill switch; audit last N hours of output logs for exposure scope. |
| **Data Exfiltration** | Disable agent tool calling; revoke external API credentials. |
| **Jailbreak** | Enable strict guardrail mode; reduce model temperature to 0. |
| **Model DoS** | Rate limit aggressive; block offending input patterns. |

### Phase 3: Investigation (< 4 hours for P0)

**Evidence Collection Checklist:**
- [ ] Full trace: user input → system prompt → retrieved context → model output.
- [ ] Guardrail decision logs (what was checked, what passed/failed).
- [ ] Tool call logs (function name, arguments, return values).
- [ ] User session history (prior turns that may have set up the attack).
- [ ] Model version and prompt template version at time of incident.

**Root Cause Analysis Questions:**
1. Was the attack in the user input or in the retrieved context (indirect injection)?
2. Did the guardrail fire and fail, or did it not fire at all?
3. Was this a known attack pattern or a novel technique?
4. Was the vulnerability introduced by a recent deployment?

### Phase 4: Remediation

| Root Cause | Fix |
| :--- | :--- |
| **Direct injection bypassed regex** | Add pattern to detection rules; consider ML-based classifier. |
| **Indirect injection via retrieved doc** | Add output-side guardrail; sanitize retrieved content before injection. |
| **PII in training data** | Audit training pipeline; add PII scrubbing to data preprocessing. |
| **Tool misuse** | Tighten tool parameter validation; add HITL for high-risk tools. |
| **Guardrail gap** | Add the specific test case to adversarial eval suite. |

### Phase 5: Recovery & Verification

1. Deploy the fix to staging.
2. Run the full adversarial evaluation suite (including the new attack vector).
3. Verify guardrail catch rate returns to baseline.
4. Gradually re-enable (canary → 10% → 50% → 100%).
5. Monitor for 24 hours post-recovery.

---

## 3. Communication Protocol

| Audience | When | Channel | Content |
| :--- | :--- | :--- | :--- |
| **On-call team** | Immediately | PagerDuty/Slack | Alert + severity + initial assessment. |
| **Security lead** | P0: < 15 min | Direct message | Full context + containment status. |
| **Engineering manager** | P0/P1: < 30 min | Slack/Email | Impact scope + ETA for fix. |
| **Legal/Compliance** | If PII exposed | Email | Scope of exposure + affected users. |
| **Affected users** | If data exposed | Per compliance requirements | Disclosure per GDPR/CCPA timelines. |

---

## 4. Adversarial Test Suite (Post-Incident Update)

Every security incident MUST result in a new test case added to the adversarial evaluation suite:

```python
# Example: Add to adversarial_eval_suite.jsonl
{
    "id": "SEC-2024-042",
    "category": "prompt_injection",
    "input": "<the actual attack input, sanitized>",
    "expected_behavior": "blocked_by_guardrail",
    "source": "production_incident",
    "date_added": "2024-12-15"
}
```

This ensures the same attack vector is automatically tested in every future deployment.

---

## 5. Quarterly Security Review Checklist

- [ ] Run full red-team exercise (internal or external).
- [ ] Review and update guardrail rules based on latest attack research.
- [ ] Audit tool permissions (principle of least privilege).
- [ ] Verify PII detection coverage against new data patterns.
- [ ] Review and rotate all API keys and credentials.
- [ ] Update this playbook with lessons from recent incidents.