"""
LLM Observability Collector — Trace, metrics, and cost tracking for production LLM apps.

Demonstrates:
1. Request-level tracing with latency, tokens, and cost.
2. Metrics aggregation (P50/P95/P99 latency, token throughput, error rate).
3. Drift detection via embedding similarity monitoring.
"""

import time
import uuid
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── 1. Trace Data Model ────────────────────────────────────────────

class SpanKind(str, Enum):
    LLM_CALL = "llm_call"
    RETRIEVAL = "retrieval"
    TOOL_CALL = "tool_call"
    RERANK = "rerank"
    GUARDRAIL = "guardrail"


@dataclass
class Span:
    """A single operation within a trace (e.g., one LLM call or one retrieval)."""
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    kind: SpanKind = SpanKind.LLM_CALL
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    status: str = "ok"  # "ok" | "error" | "timeout"
    metadata: dict = field(default_factory=dict)


@dataclass
class Trace:
    """End-to-end trace for a single user request."""
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    session_id: str = ""
    user_id: str = ""
    query: str = ""
    response: str = ""
    spans: list[Span] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0

    @property
    def total_latency_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000 if self.end_time else 0

    @property
    def total_tokens(self) -> int:
        return sum(s.input_tokens + s.output_tokens for s in self.spans)

    @property
    def total_cost_usd(self) -> float:
        """Estimate cost based on model pricing (simplified)."""
        pricing = {
            "gpt-4o": (2.50, 10.00),        # per 1M tokens (input, output)
            "gpt-4o-mini": (0.15, 0.60),
            "claude-sonnet": (3.00, 15.00),
            "claude-haiku": (0.25, 1.25),
        }
        total = 0.0
        for s in self.spans:
            if s.kind == SpanKind.LLM_CALL and s.model in pricing:
                inp_price, out_price = pricing[s.model]
                total += (s.input_tokens * inp_price + s.output_tokens * out_price) / 1_000_000
        return round(total, 6)


# ── 2. Metrics Aggregator ──────────────────────────────────────────

class MetricsAggregator:
    """Collects traces and computes operational metrics."""

    def __init__(self):
        self.traces: list[Trace] = []

    def record(self, trace: Trace):
        self.traces.append(trace)

    def latency_percentiles(self) -> dict:
        latencies = [t.total_latency_ms for t in self.traces if t.end_time > 0]
        if not latencies:
            return {"p50": 0, "p95": 0, "p99": 0}
        latencies.sort()
        n = len(latencies)
        return {
            "p50": latencies[int(n * 0.50)],
            "p95": latencies[int(n * 0.95)] if n >= 20 else latencies[-1],
            "p99": latencies[int(n * 0.99)] if n >= 100 else latencies[-1],
        }

    def error_rate(self) -> float:
        if not self.traces:
            return 0.0
        errors = sum(1 for t in self.traces if any(s.status == "error" for s in t.spans))
        return errors / len(self.traces)

    def token_throughput(self) -> dict:
        """Tokens per second across all traces."""
        total_tokens = sum(t.total_tokens for t in self.traces)
        total_seconds = sum((t.end_time - t.start_time) for t in self.traces if t.end_time > 0)
        tps = total_tokens / total_seconds if total_seconds > 0 else 0
        return {"total_tokens": total_tokens, "tokens_per_second": round(tps, 1)}

    def cost_summary(self) -> dict:
        costs = [t.total_cost_usd for t in self.traces]
        return {
            "total_usd": round(sum(costs), 4),
            "avg_per_request_usd": round(statistics.mean(costs), 6) if costs else 0,
            "max_single_request_usd": round(max(costs), 6) if costs else 0,
        }

    def report(self) -> dict:
        return {
            "total_requests": len(self.traces),
            "latency": self.latency_percentiles(),
            "error_rate": f"{self.error_rate():.2%}",
            "throughput": self.token_throughput(),
            "cost": self.cost_summary(),
        }


# ── 3. Drift Detector (Embedding Similarity) ──────────────────────

class DriftDetector:
    """
    Monitors query distribution drift by comparing embedding centroids.
    If the average similarity between recent queries and the baseline drops
    below a threshold, it signals drift.
    """

    def __init__(self, baseline_centroid: list[float], threshold: float = 0.75):
        self.baseline = baseline_centroid
        self.threshold = threshold
        self.recent_similarities: list[float] = []

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    def check(self, query_embedding: list[float]) -> dict:
        sim = self.cosine_similarity(query_embedding, self.baseline)
        self.recent_similarities.append(sim)

        # Rolling window of last 100
        window = self.recent_similarities[-100:]
        avg_sim = statistics.mean(window)
        is_drifting = avg_sim < self.threshold

        return {
            "current_similarity": round(sim, 4),
            "rolling_avg_similarity": round(avg_sim, 4),
            "is_drifting": is_drifting,
            "action": "ALERT: Query distribution drift detected" if is_drifting else "OK",
        }


# ── 4. Demo ────────────────────────────────────────────────────────

if __name__ == "__main__":
    agg = MetricsAggregator()

    # Simulate 5 requests
    for i in range(5):
        trace = Trace(user_id=f"user_{i}", query=f"Question {i}")
        trace.spans.append(Span(
            kind=SpanKind.RETRIEVAL, latency_ms=80, metadata={"docs_retrieved": 5}
        ))
        trace.spans.append(Span(
            kind=SpanKind.LLM_CALL, model="gpt-4o",
            input_tokens=1500, output_tokens=400, latency_ms=1200,
            status="error" if i == 3 else "ok",
        ))
        trace.end_time = trace.start_time + (1.28 + i * 0.1)
        agg.record(trace)

    import json as _json
    print(_json.dumps(agg.report(), indent=2))