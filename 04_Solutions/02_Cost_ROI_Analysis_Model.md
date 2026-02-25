# Cost & ROI Analysis Model for LLM Projects

*Prerequisite: [01_Technology_Selection.md](01_Technology_Selection.md).*

---

This document provides a framework for calculating the Total Cost of Ownership (TCO) and Return on Investment (ROI) for domain LLM applications, enabling data-driven business justification.

## 1. Cost Structure Overview

```
Total Cost = Development Cost + Infrastructure Cost + Operational Cost + Opportunity Cost
```

---

## 2. Development Cost (One-Time + Iterative)

| Item | Description | Typical Range |
| :--- | :--- | :--- |
| **Data Collection & Labeling** | Expert annotation, synthetic data generation, quality review. | $10K - $200K |
| **Fine-tuning Compute** | GPU hours for CPT/SFT/DPO training runs. | $1K - $50K per run |
| **RAG Infrastructure** | Vector DB setup, chunking pipeline, embedding generation. | $5K - $30K |
| **Evaluation Suite** | Golden dataset creation, LLM-as-a-Judge pipeline. | $5K - $20K |
| **Integration & UI** | API development, frontend, auth, logging. | $20K - $100K |

**Key Insight**: Data labeling and evaluation are often 40-60% of total development cost. Budget accordingly.

---

## 3. Infrastructure Cost (Recurring)

### 3.1 API-Based (OpenAI, Anthropic, etc.)

```
Monthly Cost = (Avg Input Tokens × Input Price) + (Avg Output Tokens × Output Price)
             × Requests per Month
             + Embedding API calls
             + Vector DB hosting
```

**Example Calculation (GPT-4o, 2024 pricing):**
- 10K requests/day × 2K input tokens × $2.50/1M = $50/day input
- 10K requests/day × 500 output tokens × $10.00/1M = $50/day output
- Monthly API cost ≈ **$3,000**
- Vector DB (managed) ≈ **$200-500/month**

**Cost Optimization Levers:**
- **Prompt Caching**: 50-90% reduction on repeated system prompts.
- **Smaller Models for Routing**: Use GPT-4o-mini or Haiku for classification/routing, reserve large models for generation.
- **Batch API**: 50% discount for non-real-time workloads.

### 3.2 Self-Hosted (vLLM, TGI, etc.)

| Component | Spec | Monthly Cost (Cloud) |
| :--- | :--- | :--- |
| **GPU Instance** | 1× A100 80GB (or H100) | $2,000 - $4,000 |
| **CPU + RAM** | 32 vCPU, 128GB RAM | $500 - $800 |
| **Storage** | 1TB SSD (model weights + index) | $100 - $200 |
| **Networking** | Load balancer + egress | $100 - $300 |

**Break-even Analysis**: Self-hosting typically becomes cheaper than API at **>50K requests/day** for a 70B model, or **>200K requests/day** for a 7B model. Below that, API is more cost-effective.

---

## 4. Operational Cost (Recurring)

| Item | Description | Monthly Range |
| :--- | :--- | :--- |
| **Monitoring & Observability** | LangSmith/Langfuse, Prometheus, Grafana. | $200 - $2,000 |
| **Human Review** | Expert review of edge cases, HITL approvals. | $2,000 - $10,000 |
| **Retraining / Eval Cycles** | Monthly re-evaluation and potential re-tuning. | $500 - $5,000 |
| **Incident Response** | On-call for model failures, hallucination incidents. | Included in team cost |

---

## 5. ROI Calculation Framework

### 5.1 Value Drivers

| Value Type | Metric | How to Measure |
| :--- | :--- | :--- |
| **Time Savings** | Hours saved per task × hourly cost of employee. | A/B test: with vs without LLM. |
| **Throughput Increase** | Additional tasks completed per unit time. | Before/after deployment comparison. |
| **Quality Improvement** | Error rate reduction, consistency score. | Expert blind review of outputs. |
| **Revenue Enablement** | New capabilities that were previously impossible. | Customer willingness-to-pay surveys. |

### 5.2 ROI Formula

```
ROI = (Annual Value Generated - Annual Total Cost) / Annual Total Cost × 100%

Where:
  Annual Value = (Time Saved × Hourly Rate × 12)
               + (Revenue from New Capabilities)
               + (Cost of Errors Avoided)

  Annual Total Cost = Development Cost (amortized over 2-3 years)
                    + Infrastructure Cost × 12
                    + Operational Cost × 12
```

### 5.3 Example: Legal Contract Review

| Metric | Before LLM | After LLM |
| :--- | :--- | :--- |
| Time per contract review | 4 hours | 0.5 hours |
| Reviews per month | 200 | 200 |
| Lawyer hourly rate | $150 | $150 |
| Monthly labor cost | $120,000 | $15,000 |
| Monthly LLM cost | $0 | $8,000 |
| **Monthly net savings** | — | **$97,000** |
| **Annual ROI** | — | **~800%** (after $150K dev cost) |

---

## 6. Decision Matrix: Build vs Buy vs Hybrid

| Factor | API (Buy) | Self-Host (Build) | Hybrid |
| :--- | :--- | :--- | :--- |
| **Data Privacy** | Data leaves your network | Full control | Sensitive via self-host, rest via API |
| **Upfront Cost** | Near zero | High (GPU procurement) | Medium |
| **Scaling** | Instant | Manual (add GPUs) | Flexible |
| **Customization** | Prompt-only | Full fine-tuning | Fine-tune core, API for auxiliary |
| **Vendor Lock-in** | High | None | Low |
| **Best For** | PoC, low-volume, fast iteration | High-volume, regulated industries | Most production scenarios |

---

## 7. Cost Monitoring Checklist

- [ ] **Per-request cost tracking**: Log token counts and model used for every request.
- [ ] **Budget alerts**: Set daily/weekly spending caps with automatic notifications.
- [ ] **Model tier optimization**: Weekly review of which requests could use a cheaper model.
- [ ] **Cache hit rate**: Monitor prompt caching effectiveness (target: >60% for repetitive workloads).
- [ ] **Idle resource detection**: Alert on GPU instances with <30% utilization.

---

## Key References

1. **Chen et al. (2023)**: *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance*.
