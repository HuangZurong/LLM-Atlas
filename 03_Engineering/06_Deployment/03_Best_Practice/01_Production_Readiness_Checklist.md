# Production Readiness Checklist

*Prerequisite: [../01_Theory/02_Deployment_Architecture.md](../01_Theory/02_Deployment_Architecture.md).*

---

Before deploying an LLM inference service to production, verify all items in this checklist. This is based on lessons from production incidents at scale.

## 1. Infrastructure

### 1.1 Kubernetes & Orchestration

- [ ] **Node Pools**: Dedicated GPU node pools separate from CPU workloads
- [ ] **GPU Operator**: NVIDIA GPU operator installed and configured
- [ ] **Resource Limits**: CPU, memory, GPU limits set for all containers
- [ ] **Priority Classes**: Critical workloads marked with higher priority
- [ ] **Pod Disruption Budget**: Max 1 unavailable pod for rolling updates
- [ ] **Node Affinity**: GPU pods pinned to GPU nodes
- [ ] **Pod Anti-Affinity**: Pods spread across nodes for high availability
- [ ] **Tolerations**: GPU workloads tolerate `nvidia.com/gpu:NoSchedule`
- [ ] **Namespace Isolation**: Separate namespaces for staging/production

### 1.2 Storage & Networking

- [ ] **Model Cache**: Persistent volume for HuggingFace cache (500GB+)
- [ ] **Network Policies**: Ingress/egress traffic restricted
- [ ] **Service Mesh**: Istio/Linkerd for traffic management (optional)
- [ ] **Load Balancer**: External load balancer with health checks
- [ ] **DNS**: Internal service discovery configured
- [ ] **TLS**: SSL termination at ingress, mTLS for internal traffic

### 1.3 Monitoring Stack

- [ ] **Prometheus**: Metrics collection with GPU-specific exporters
- [ ] **Grafana**: Dashboards for latency, throughput, GPU utilization
- [ ] **AlertManager**: Alerts configured for SLO violations
- [ ] **Log Aggregation**: ELK/Loki stack for centralized logs
- [ ] **Distributed Tracing**: Jaeger/Tempo with OpenTelemetry
- [ ] **Synthetic Monitoring**: External health checks from multiple regions

## 2. Application

### 2.1 Inference Engine

- [ ] **Engine Selection**: vLLM/TGI/SGLang chosen based on requirements
- [ ] **Quantization**: Model quantized to appropriate precision (8-bit/4-bit)
- [ ] **Continuous Batching**: Enabled for high throughput
- [ ] **Prefix Caching**: Enabled if using standardized prompts
- [ ] **Max Concurrency**: Limits set to prevent OOM (e.g., max 256 sequences)
- [ ] **Graceful Shutdown**: SIGTERM handling with request draining
- [ ] **Health Endpoints**: `/health`, `/ready`, `/metrics` endpoints
- [ ] **Request Validation**: Input validation and size limits

### 2.2 API Design

- [ ] **REST Interface**: Standardized request/response formats
- [ ] **Streaming Support**: SSE for real-time token delivery
- [ ] **Rate Limiting**: Per-user/org rate limits
- [ ] **Authentication**: API keys, JWT, or OAuth
- [ ] **Request ID**: Unique ID per request for tracing
- [ ] **CORS**: Cross-origin configuration for web clients
- [ ] **Versioning**: API version in URL or header

### 2.3 Performance & Reliability

- [ ] **Circuit Breakers**: Fail-fast when downstream services fail
- [ ] **Retry Logic**: Exponential backoff for transient failures
- [ ] **Timeout Configuration**: Per-request and overall timeouts
- [ ] **Connection Pooling**: HTTP/DB connection reuse
- [ ] **Caching Layers**:
  - L1: In-memory (Redis) for frequent queries
  - L2: Semantic cache for similar queries
  - L3: Exact match cache for identical queries
- [ ] **Load Shedding**: Drop low-priority requests under high load

## 3. Model Management

### 3.1 Deployment Pipeline

- [ ] **Model Registry**: Centralized model storage (MLflow, DVC, S3)
- [ ] **Versioning**: Semantic versioning for models
- [ ] **A/B Testing**: Canary deployments with traffic splitting
- [ ] **Shadow Mode**: New models process traffic without affecting users
- [ ] **Rollback Plan**: Automated rollback on quality regression
- [ ] **Model Warm-up**: Pre-load models before traffic routing
- [ ] **Dependency Tracking**: Model -> code -> data lineage

### 3.2 Quality Assurance

- [ ] **Golden Dataset**: 100+ representative Q&A pairs
- [ ] **Automated Testing**: Unit tests for critical paths
- [ ] **Integration Testing**: End-to-end tests with real models
- [ ] **Performance Testing**: Load testing at 2x expected traffic
- [ ] **Regression Testing**: Quality metrics tracked per commit
- [ ] **Security Scanning**: SAST/DAST on model and code
- [ ] **Compliance Checks**: PII detection, content safety

## 4. Observability

### 4.1 Metrics

- [ ] **Business Metrics**:
  - Requests per second (RPS)
  - Success rate (5xx errors < 1%)
  - Average latency (p50 < 500ms)
  - Tail latency (p95 < 2s, p99 < 5s)

- [ ] **Model Metrics**:
  - Time to first token (TTFT) < 300ms
  - Tokens per second (TPS) > 100
  - GPU utilization (target 70-90%)
  - VRAM usage (alert at >90%)

- [ ] **Cost Metrics**:
  - Cost per request ($)
  - Tokens per dollar (efficiency)
  - Cache hit rate (>30% target)

### 4.2 Logging

- [ ] **Structured Logs**: JSON format with consistent fields
- [ ] **Request/Response Logs**: Full request/response for debugging
- [ ] **Error Classification**: Different log levels for different errors
- [ ] **Sensitive Data**: PII masked in logs
- [ ] **Trace ID**: Correlation ID across all services
- [ ] **Retention Policy**: 30 days hot, 1 year cold storage

### 4.3 Tracing

- [ ] **Distributed Tracing**: Full request flow across services
- [ ] **Span Attributes**: Model name, token count, latency
- [ ] **Sampling Strategy**: 100% for errors, 10% for successful requests
- [ ] **Trace Export**: To centralized backend (Jaeger/Tempo)

## 5. Security

### 5.1 Authentication & Authorization

- [ ] **API Authentication**: API keys, JWT, or OAuth 2.0
- [ ] **Rate Limiting**: Per-user, per-organization quotas
- [ ] **Access Control**: RBAC for different user types
- [ ] **Audit Logging**: All API calls logged with user context

### 5.2 Data Protection

- [ ] **PII Detection**: Scan inputs/outputs for sensitive data
- [ ] **Data Encryption**: At rest and in transit
- [ ] **Data Retention**: Clear policies for request/response storage
- [ ] **Data Sovereignty**: GDPR/CCPA compliance if applicable

### 5.3 Model Security

- [ ] **Prompt Injection**: Input sanitization and detection
- [ ] **Model Evasion**: Testing for adversarial examples
- [ ] **Output Safety**: Content moderation on generated text
- [ ] **Tool Safety**: Sandboxed execution for code generation

## 6. Operations

### 6.1 Deployment

- [ ] **Blue-Green Deployments**: Zero-downtime updates
- [ ] **Canary Releases**: Gradual traffic shift to new versions
- [ ] **Rollback Automation**: One-click rollback to previous version
- [ ] **Deployment Windows**: Off-peak hours for risky changes

### 6.2 Incident Response

- [ ] **Runbooks**: Documented procedures for common incidents
- [ ] **On-call Rotation**: 24/7 coverage with escalation paths
- [ ] **Incident Classification**: P0-P4 severity levels
- [ ] **Post-mortems**: Blameless analysis of all P0/P1 incidents
- [ ] **Chaos Engineering**: Regular failure injection tests

### 6.3 Capacity Planning

- [ ] **Load Forecasting**: Predict traffic growth (daily/weekly/seasonal)
- [ ] **Auto-scaling**: Horizontal scaling based on metrics
- [ ] **Resource Reservation**: Buffer capacity for spikes (20-30%)
- [ ] **Cost Optimization**: Spot instances for batch, reserved for steady-state

## 7. Compliance & Governance

### 7.1 Regulatory Compliance

- [ ] **SOC2**: Security controls documented and tested
- [ ] **GDPR**: Data protection for EU citizens
- [ ] **HIPAA**: If handling healthcare data
- [ ] **PCI DSS**: If handling payment data

### 7.2 Internal Governance

- [ ] **Change Management**: All changes go through PR review
- [ ] **Access Reviews**: Quarterly review of who has access
- [ ] **Security Scanning**: Regular vulnerability scans
- [ ] **Backup Strategy**: Regular backups of critical data

## 8. Documentation

### 8.1 Technical Documentation

- [ ] **Architecture Diagrams**: System design and data flow
- [ ] **API Documentation**: OpenAPI/Swagger specs
- [ ] **Runbooks**: Step-by-step operational procedures
- [ ] **Troubleshooting Guide**: Common issues and solutions
- [ ] **Capacity Planning Guide**: How to scale the system

### 8.2 User Documentation

- [ ] **Getting Started Guide**: Hello world example
- [ ] **Best Practices**: How to use the API effectively
- [ ] **Rate Limits**: Clear documentation of limits
- [ ] **Error Codes**: All possible errors and resolutions
- [ ] **Support Channels**: How to get help

## Verification

Before production launch, conduct these final checks:

1. **Load Test**: 2x expected peak traffic for 1 hour
2. **Chaos Test**: Kill random pods, network partitions
3. **Rollback Test**: Deploy → Rollback → Verify recovery
4. **Security Audit**: External penetration test
5. **Documentation Review**: All docs up-to-date
6. **Training**: Operations team trained on new system

**Go/No-Go Decision**: All items must be checked or have approved exceptions before production launch.