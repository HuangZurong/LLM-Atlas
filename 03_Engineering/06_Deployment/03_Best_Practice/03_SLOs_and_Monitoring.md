# SLOs and Monitoring for LLM Serving

*Prerequisite: [01_Production_Readiness_Checklist.md](01_Production_Readiness_Checklist.md).*

---

In LLM deployment, traditional web metrics (uptime, status codes) are insufficient. You must track token-level metrics and GPU health to ensure a quality user experience.

## 1. Defining Service Level Objectives (SLOs)

| Metric | SLO Target (Tier 1) | SLO Target (Tier 2) | Rationale |
| :--- | :--- | :--- | :--- |
| **Availability** | 99.9% | 99.5% | Standard uptime requirement |
| **TTFT (Time to First Token)** | < 300ms | < 800ms | Directly impacts perceived speed |
| **TPS (Tokens Per Second)** | > 50 tokens/s | > 20 tokens/s | Reading speed is ~5-10 tokens/s |
| **Error Rate (5xx)** | < 0.1% | < 1.0% | Measures stability |
| **GPU Utilization** | 70-85% | 50-70% | Efficiency vs. Headroom balance |

## 2. Key Performance Indicators (KPIs)

### 2.1 Latency Metrics
- **TTFT**: Time from request arrival to the first token being generated.
- **TPOT (Time Per Output Token)**: Average time taken to generate each subsequent token.
- **E2E Latency**: Total time for the entire request completion.

### 2.2 Throughput Metrics
- **Requests Per Second (RPS)**: Number of concurrent users served.
- **Tokens Per Second (TPS)**: Total aggregate throughput across all active batches.

### 2.3 Efficiency Metrics
- **Batching Efficiency**: Average batch size vs. max batch size.
- **KV Cache Utilization**: Percentage of allocated VRAM used for caching.

## 3. Monitoring Infrastructure

### 3.1 Exporters
- **NVIDIA Data Center GPU Manager (DCGM)**: For GPU temperature, power, and utilization.
- **vLLM/TGI Metrics**: Native Prometheus endpoints for token stats and queue depth.
- **Prometheus Node Exporter**: For CPU, RAM, and disk IO.

### 3.2 Dashboard Patterns
A production LLM dashboard should include four primary views:
1. **Executive View**: Cost per 1k tokens, total usage, global availability.
2. **Operations View**: GPU memory usage, pod restarts, error logs.
3. **Engineering View**: TTFT/TPS histograms, batching efficiency, cache hit rate.
4. **Model View**: Comparison of metrics across different deployed models (A/B testing).

## 4. Alerting Strategies

- **P0 Alert**: Any GPU node goes `Unhealthy` or `OOM`.
- **P1 Alert**: TTFT (p95) exceeds 2 seconds for more than 5 minutes.
- **P1 Alert**: Success rate drops below 98%.
- **P2 Alert**: GPU utilization exceeds 90% (indicates need for auto-scaling).

## 5. Tooling
- **Prometheus**: Time-series database for metrics.
- **Grafana**: Visualization.
- **Loki / ELK**: Log aggregation.
- **LangSmith / Weights & Biases**: For tracing and evaluation quality.
