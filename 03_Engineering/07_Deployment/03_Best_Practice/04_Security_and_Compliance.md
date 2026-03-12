# Security & Compliance for LLM Deployment

*Prerequisite: [01_Production_Readiness_Checklist.md](01_Production_Readiness_Checklist.md).*

---

Deploying LLMs involves risks ranging from prompt injection to data leaks. This document outlines the security controls required for production-grade model serving.

## 1. Authentication and Authorization

- **API Keys**: Use scoped API keys for different clients/features.
- **JWT / OAuth2**: Standard for user-facing applications.
- **Role-Based Access Control (RBAC)**: Limit which users can access specific models (e.g., GPT-4 reserved for internal staff).

## 2. Infrastructure Security

- **VPC Isolation**: Deploy inference engines within a private VPC.
- **mTLS**: Ensure all traffic between the API gateway and the inference engine is encrypted.
- **GPU Isolation**: In multi-tenant environments, use MIG (Multi-Instance GPU) or Kubernetes namespaces to isolate workloads.

## 3. Data Privacy & Compliance

### 3.1 PII Protection
- **Detection**: Use models like Presidio to identify PII in inputs/outputs.
- **Redaction**: Mask sensitive data (emails, credit cards) before logging or sending to external APIs.

### 3.2 Regulatory Alignment
- **GDPR**: Ensure data is processed in the correct geographic region.
- **Zero-Retention**: If using 3rd party APIs, opt into enterprise zero-data-retention policies.

## 4. Prompt Injection Mitigation

- **System Prompt Hardening**: Clear separation between instructions and data.
- **Input Sanitization**: Block common injection patterns (e.g., "ignore all previous instructions").
- **Output Validation**: Ensure the model hasn't been tricked into outputting forbidden code or system secrets.

## 5. Monitoring and Audit

- **Audit Logs**: Record user ID, request timestamp, model used, and token count.
- **Anomaly Detection**: Monitor for unusual traffic patterns that might indicate model scraping or prompt injection attacks.
- **Human-in-the-loop (HITL)**: For high-stakes deployments, log and sample model outputs for manual security review.
