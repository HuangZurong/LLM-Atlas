# Cloud Platform Comparison: AI/ML Deployment

*Prerequisite: [02_Deployment_Architecture.md](02_Deployment_Architecture.md).*

---

Choosing where to deploy your LLM application is as critical as choosing the model itself. This document compares the major cloud platforms for AI inference, including emerging GPU cloud providers.

## 1. The AI Cloud Landscape (2025)

### 1.1 Market Segments

```
┌─────────────────────────────────────────────────┐
│ Tier 1: Hyper-scalers (AWS, Azure, GCP)         │
│   - Full ecosystem integration                  │
│   - Enterprise SLAs, compliance                 │
│   - Highest cost, most managed                  │
├─────────────────────────────────────────────────┤
│ Tier 2: Specialized GPU Clouds (CoreWeave, etc.)│
│   - Raw GPU power, competitive pricing          │
│   - NVIDIA partnership benefits                 │
│   - Less ecosystem, more technical              │
├─────────────────────────────────────────────────┤
│ Tier 3: Inference-as-a-Service (OpenAI, etc.)   │
│   - No infrastructure management                │
│   - Pay-per-token, simple scaling               │
│   - Vendor lock-in, limited control             │
└─────────────────────────────────────────────────┘
```

### 1.2 Decision Framework

```
Start
  ├─ Need enterprise compliance (HIPAA, SOC2)? ──Yes──> AWS/Azure/GCP
  │
  ├─ Need raw GPU power at lowest cost? ──Yes──> CoreWeave/Lambda
  │
  ├─ Want zero infrastructure management? ──Yes──> OpenAI API / Anthropic
  │
  ├─ Already heavily invested in a cloud? ──Yes──> Stay there
  │
  └─ Default: GCP (best AI tooling) or CoreWeave (best price/performance)
```

## 2. Platform Deep Dive

### 2.1 AWS SageMaker

**Strengths**:
- **Deep ecosystem**: Integrates with S3, CloudWatch, IAM, VPC
- **Enterprise-ready**: HIPAA, SOC2, PCI DSS compliance
- **Inference optimization**: SageMaker Neo compiles models for specific hardware
- **Multi-model endpoints**: Serve multiple models on a single endpoint

**Weaknesses**:
- **Complex pricing**: Instance hours + SageMaker fees + data transfer
- **GPU availability**: Spot instances unreliable for production
- **Learning curve**: Many moving parts

**Best for**: Enterprises already on AWS needing compliance and ecosystem integration.

**Pricing Example (US East)**:
- `ml.g5.12xlarge` (4×A10G, 96GB VRAM): $5.67/hour
- `ml.p4d.24xlarge` (8×A100, 320GB VRAM): $32.77/hour
- **SageMaker fee**: +$0.10/hour per endpoint

### 2.2 Azure Machine Learning

**Strengths**:
- **Microsoft integration**: Active Directory, Power BI, Office 365
- **Azure OpenAI Service**: Managed GPT-4/3.5 with enterprise controls
- **MLOps pipelines**: Robust CI/CD for machine learning
- **Managed online endpoints**: Auto-scaling with GPU support

**Weaknesses**:
- **Region variability**: GPU availability varies by region
- **Portal complexity**: Multiple portals (Azure Portal, ML Studio)
- **Cost visibility**: Complex billing breakdowns

**Best for**: Microsoft shops, enterprises using Azure OpenAI.

**Pricing Example (East US)**:
- `Standard_NC24ads_A100_v4` (1×A100, 80GB): $32.77/hour
- `Standard_NC96ads_A100_v4` (8×A100, 640GB): $262.16/hour
- **Managed endpoint**: +20% surcharge

### 2.3 Google Cloud Platform (Vertex AI)

**Strengths**:
- **TPU-native**: Best for TPU-optimized models (PaLM, Gemma)
- **Vertex AI Prediction**: Auto-scaling with traffic-based pricing
- **Model Registry**: Centralized model management
- **Custom containers**: Full Docker flexibility

**Weaknesses**:
- **GPU selection**: Limited compared to AWS
- **Region constraints**: Fewer regions with high-end GPUs
- **Documentation gaps**: Some newer features under-documented

**Best for**: Teams using Google models (Gemini, Gemma), TPU workloads.

**Pricing Example (US Central)**:
- `a2-highgpu-1g` (1×A100, 40GB): $2.93/hour
- `a2-ultragpu-8g` (8×A100, 320GB): $23.44/hour
- **Prediction traffic**: $0.0001 per 1K characters

### 2.4 CoreWeave

**Strengths**:
- **Raw GPU power**: Access to H100, A100, L40S at scale
- **Competitive pricing**: Often 30-50% cheaper than hyperscalers
- **NVIDIA partnership**: Early access to new hardware
- **Simple pricing**: Hourly rates, no complex tiering

**Weaknesses**:
- **Limited ecosystem**: Fewer managed services
- **Network latency**: Fewer global regions
- **Enterprise features**: Less mature compliance offerings

**Best for**: AI startups, research labs, cost-sensitive production.

**Pricing Example**:
- `H100-80GB-PCIe` (1×H100): $4.76/hour
- `A100-80GB-PCIe` (1×A100): $2.76/hour
- `RTX 4090` (consumer GPU): $0.79/hour

### 2.5 Lambda Labs / RunPod

**Strengths**:
- **Spot-like pricing**: Bid for excess capacity
- **Quick start**: GPU instances in minutes
- **Community support**: Popular with researchers
- **Jupyter integration**: Built-in notebooks

**Weaknesses**:
- **Reliability**: Instances can be preempted
- **Limited support**: Community vs. enterprise support
- **Security**: Less mature isolation

**Best for**: Experimentation, development, non-critical workloads.

## 3. Platform Comparison Matrix

| Feature | AWS | Azure | GCP | CoreWeave | Lambda |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **H100 Availability** | Limited | Limited | Limited | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **A100 Pricing (80GB/hr)** | $32.77 | $32.77 | $23.44 | $2.76 | $1.10 |
| **Auto-scaling** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ |
| **Enterprise SLA** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
| **MLOps Tooling** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐ |
| **GPU Selection** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Ecosystem Integration** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐ |
| **Cost Predictability** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |

## 4. Inference-as-a-Service (IaaS)

### 4.1 When to Use API Services

| Scenario | Recommended Service |
| :--- | :--- |
| **Prototyping / MVP** | OpenAI API, Anthropic Claude |
| **Enterprise chat** | Azure OpenAI (compliance) |
| **High-availability** | Multiple providers + fallback |
| **Cost-sensitive** | Self-host on CoreWeave |

### 4.2 API vs. Self-hosted Cost Comparison

**Example**: 10M tokens/month (mix of input/output)

| Option | Monthly Cost | Latency | Control |
| :--- | :--- | :--- | :--- |
| **OpenAI GPT-4o** | $2,500 | 200-500ms | Low |
| **Anthropic Claude 3.5 Sonnet** | $3,000 | 300-700ms | Low |
| **Self-hosted Llama-3.1-70B (A100)** | $1,980 | 100-300ms | High |
| **Self-hosted Qwen2.5-32B (A10)** | $890 | 200-400ms | High |

**Break-even point**: Self-hosting becomes cheaper at ~5M tokens/month for 70B models.

## 5. Hybrid Deployment Strategies

### 5.1 Tiered Inference

```
User Request → Router
    ├─ Simple query → Cheap model (7B, self-hosted)
    ├─ Medium complexity → Medium model (32B, self-hosted)
    ├─ High complexity → GPT-4o (API fallback)
    └─ Critical mission → Ensemble (multiple models)
```

### 5.2 Multi-Cloud Resilience

```yaml
# Example: Multi-cloud failover configuration
inference_providers:
  primary:
    provider: "coreweave"
    region: "us-east"
    model: "llama-3.1-70b"

  secondary:
    provider: "aws"
    region: "us-east-1"
    model: "llama-3.1-70b"

  fallback:
    provider: "openai"
    model: "gpt-4o"
```

### 5.3 Cost Optimization Patterns

1. **Spot/Preemptible instances** for batch inference (save 70-90%)
2. **Warm pool** of instances for consistent latency
3. **Region arbitrage** — deploy in cheapest regions with acceptable latency
4. **Reserved instances** for predictable workloads (save 30-40%)

## 6. Migration Considerations

### 6.1 Portability Checklist

- [ ] **Model format**: Use standard formats (GGUF, SafeTensors)
- [ ] **Inference engine**: Choose portable engines (vLLM, TGI)
- [ ] **Configuration**: Environment variables, not hardcoded paths
- [ ] **Monitoring**: Use OpenTelemetry for vendor-agnostic observability
- [ ] **Secret management**: External secrets (HashiCorp Vault, AWS Secrets Manager)

### 6.2 Exit Strategy

Always design with **cloud exit** in mind:
1. Keep data egress costs low (store data in portable formats)
2. Use containerized deployments (Docker)
3. Maintain parallel deployments (test new cloud while old runs)
4. Document migration procedures

## 7. Recommendations

1. **Startups**: CoreWeave for cost, add hyperscaler for compliance when needed
2. **Enterprises**: Stay with existing cloud provider, use managed services
3. **Research**: Lambda/RunPod for experimentation, migrate to production platform
4. **Global apps**: Multi-region on hyperscalers, edge caching for latency

**The rule**: No single platform is best for all use cases. Design for portability and use the right tool for each workload.