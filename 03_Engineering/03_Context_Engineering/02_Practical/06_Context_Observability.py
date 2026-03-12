"""
Context Observability
======================
Instruments context assembly with token usage tracking, cost attribution,
context diff logging, and OpenTelemetry-compatible span recording.

Use this alongside the other Practical files to gain visibility into what
your application is putting into the context window and what it costs.

Prerequisites:
  - 02_Practical/01_Context_Composition_Pipeline.py
  - 02_Practical/02_Token_Budget_Controller.py
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)

_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


# ---------------------------------------------------------------------------
# Pricing table (USD per 1M tokens, as of early 2025)
# Update from provider docs as prices change.
# ---------------------------------------------------------------------------

PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00, "cached_input": 0.30},
    "claude-haiku-3-5":  {"input": 0.80, "output": 4.00,  "cached_input": 0.08},
    "gpt-4o":            {"input": 2.50, "output": 10.00, "cached_input": 1.25},
    "gpt-4o-mini":       {"input": 0.15, "output": 0.60,  "cached_input": 0.075},
}


def token_cost(
    tokens: int,
    model: str,
    token_type: str = "input",  # "input" | "output" | "cached_input"
) -> float:
    """Return cost in USD for a given token count."""
    price_per_million = PRICING.get(model, {}).get(token_type, 0.0)
    return tokens * price_per_million / 1_000_000


# ---------------------------------------------------------------------------
# Context snapshot — one per LLM call
# ---------------------------------------------------------------------------

@dataclass
class ContextSnapshot:
    """Records the token composition of a single LLM call."""
    request_id: str
    session_id: str
    turn: int
    model: str
    timestamp: float = field(default_factory=time.time)

    # Layer token counts
    layer_tokens: dict[str, int] = field(default_factory=dict)

    # Cache info
    cached_tokens: int = 0
    cache_hit: bool = False

    # Output
    output_tokens: int = 0

    # Assembly metadata
    compression_applied: bool = False
    layers_trimmed: list[str] = field(default_factory=list)
    assembly_latency_ms: float = 0.0

    @property
    def total_input_tokens(self) -> int:
        return sum(self.layer_tokens.values())

    @property
    def uncached_input_tokens(self) -> int:
        return max(0, self.total_input_tokens - self.cached_tokens)

    @property
    def total_cost_usd(self) -> float:
        return (
            token_cost(self.uncached_input_tokens, self.model, "input")
            + token_cost(self.cached_tokens, self.model, "cached_input")
            + token_cost(self.output_tokens, self.model, "output")
        )

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "turn": self.turn,
            "model": self.model,
            "timestamp": self.timestamp,
            "layer_tokens": self.layer_tokens,
            "total_input_tokens": self.total_input_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_hit": self.cache_hit,
            "output_tokens": self.output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 8),
            "compression_applied": self.compression_applied,
            "layers_trimmed": self.layers_trimmed,
            "assembly_latency_ms": round(self.assembly_latency_ms, 2),
        }


# ---------------------------------------------------------------------------
# Context diff — what changed between consecutive turns
# ---------------------------------------------------------------------------

@dataclass
class ContextDiff:
    """Records what changed in the context between two consecutive turns."""
    session_id: str
    from_turn: int
    to_turn: int
    token_delta: int
    layer_deltas: dict[str, int]  # layer_name → token change
    compression_applied: bool
    layers_trimmed: list[str]

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "from_turn": self.from_turn,
            "to_turn": self.to_turn,
            "token_delta": self.token_delta,
            "layer_deltas": self.layer_deltas,
            "compression_applied": self.compression_applied,
            "layers_trimmed": self.layers_trimmed,
        }


# ---------------------------------------------------------------------------
# Context observer — collects and analyzes snapshots
# ---------------------------------------------------------------------------

class ContextObserver:
    """
    Collects context snapshots, computes diffs, and generates reports.

    Usage:
        observer = ContextObserver(model="claude-sonnet-4-5", context_window=128_000)

        # Before each LLM call:
        snapshot = observer.record(
            request_id="req_001",
            session_id="sess_abc",
            turn=1,
            layer_tokens={"system": 500, "rag": 3200, "history": 2800, "query": 150},
            cached_tokens=500,
            output_tokens=320,
            assembly_latency_ms=45.2,
        )

        # Get reports:
        observer.print_report()
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        context_window: int = 128_000,
        output_reserve_ratio: float = 0.25,
        alert_thresholds: dict[str, float] | None = None,
    ) -> None:
        self.model = model
        self.context_window = context_window
        self.input_budget = int(context_window * (1 - output_reserve_ratio))
        self.alert_thresholds = alert_thresholds or {
            "utilization": 0.80,          # Alert if input > 80% of budget
            "cache_hit_rate": 0.30,        # Alert if cache hit rate < 30%
            "avg_relevance": 0.60,         # Alert if avg relevance score < 0.6
            "output_truncation_rate": 0.05, # Alert if >5% of responses truncated
        }

        self._snapshots: list[ContextSnapshot] = []
        self._session_last: dict[str, ContextSnapshot] = {}  # session_id → last snapshot

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        request_id: str,
        session_id: str,
        turn: int,
        layer_tokens: dict[str, int],
        cached_tokens: int = 0,
        output_tokens: int = 0,
        compression_applied: bool = False,
        layers_trimmed: list[str] | None = None,
        assembly_latency_ms: float = 0.0,
    ) -> ContextSnapshot:
        snapshot = ContextSnapshot(
            request_id=request_id,
            session_id=session_id,
            turn=turn,
            model=self.model,
            layer_tokens=layer_tokens,
            cached_tokens=cached_tokens,
            cache_hit=cached_tokens > 0,
            output_tokens=output_tokens,
            compression_applied=compression_applied,
            layers_trimmed=layers_trimmed or [],
            assembly_latency_ms=assembly_latency_ms,
        )
        self._snapshots.append(snapshot)

        # Compute and log diff if we have a previous snapshot for this session
        if session_id in self._session_last:
            diff = self._compute_diff(self._session_last[session_id], snapshot)
            if abs(diff.token_delta) > 500:
                logger.info(
                    "Session %s turn %d→%d: %+d tokens %s",
                    session_id, diff.from_turn, diff.to_turn, diff.token_delta,
                    "(compressed)" if diff.compression_applied else "",
                )

        self._session_last[session_id] = snapshot
        self._check_alerts(snapshot)
        return snapshot

    # ------------------------------------------------------------------
    # Diff computation
    # ------------------------------------------------------------------

    def _compute_diff(self, prev: ContextSnapshot, curr: ContextSnapshot) -> ContextDiff:
        all_layers = set(prev.layer_tokens) | set(curr.layer_tokens)
        layer_deltas = {
            layer: curr.layer_tokens.get(layer, 0) - prev.layer_tokens.get(layer, 0)
            for layer in all_layers
            if curr.layer_tokens.get(layer, 0) != prev.layer_tokens.get(layer, 0)
        }
        return ContextDiff(
            session_id=curr.session_id,
            from_turn=prev.turn,
            to_turn=curr.turn,
            token_delta=curr.total_input_tokens - prev.total_input_tokens,
            layer_deltas=layer_deltas,
            compression_applied=curr.compression_applied,
            layers_trimmed=curr.layers_trimmed,
        )

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    def _check_alerts(self, snapshot: ContextSnapshot) -> None:
        utilization = snapshot.total_input_tokens / self.input_budget
        if utilization > self.alert_thresholds["utilization"]:
            logger.warning(
                "HIGH UTILIZATION: request %s at %.1f%% of input budget (%d/%d tokens)",
                snapshot.request_id, utilization * 100,
                snapshot.total_input_tokens, self.input_budget,
            )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        if not self._snapshots:
            return {}

        total_requests = len(self._snapshots)
        total_input = sum(s.total_input_tokens for s in self._snapshots)
        total_output = sum(s.output_tokens for s in self._snapshots)
        total_cached = sum(s.cached_tokens for s in self._snapshots)
        total_cost = sum(s.total_cost_usd for s in self._snapshots)
        cache_hits = sum(1 for s in self._snapshots if s.cache_hit)
        compressed = sum(1 for s in self._snapshots if s.compression_applied)

        # Per-layer totals
        layer_totals: dict[str, int] = defaultdict(int)
        for s in self._snapshots:
            for layer, tokens in s.layer_tokens.items():
                layer_totals[layer] += tokens

        avg_input = total_input / total_requests
        avg_utilization = avg_input / self.input_budget

        return {
            "total_requests": total_requests,
            "avg_input_tokens": round(avg_input),
            "avg_utilization": f"{avg_utilization:.1%}",
            "cache_hit_rate": f"{cache_hits / total_requests:.1%}",
            "compression_rate": f"{compressed / total_requests:.1%}",
            "total_cost_usd": round(total_cost, 4),
            "avg_cost_per_request_usd": round(total_cost / total_requests, 6),
            "layer_token_totals": dict(sorted(layer_totals.items(), key=lambda x: -x[1])),
            "alerts": self._generate_alerts(avg_utilization, cache_hits / total_requests),
        }

    def _generate_alerts(self, avg_utilization: float, cache_hit_rate: float) -> list[str]:
        alerts = []
        if avg_utilization > self.alert_thresholds["utilization"]:
            alerts.append(f"Avg utilization {avg_utilization:.1%} exceeds threshold — review injection logic")
        if cache_hit_rate < self.alert_thresholds["cache_hit_rate"]:
            alerts.append(f"Cache hit rate {cache_hit_rate:.1%} below threshold — check static-first ordering")
        return alerts

    def cost_attribution(self) -> dict[str, dict]:
        """Break down total cost by context layer."""
        layer_costs: dict[str, float] = defaultdict(float)
        layer_tokens: dict[str, int] = defaultdict(int)

        for snapshot in self._snapshots:
            for layer, tokens in snapshot.layer_tokens.items():
                layer_tokens[layer] += tokens
                layer_costs[layer] += token_cost(tokens, self.model, "input")

        total_cost = sum(layer_costs.values())
        return {
            layer: {
                "total_tokens": layer_tokens[layer],
                "total_cost_usd": round(layer_costs[layer], 6),
                "cost_fraction": f"{layer_costs[layer] / total_cost:.1%}" if total_cost > 0 else "0%",
            }
            for layer in sorted(layer_costs, key=lambda x: -layer_costs[x])
        }

    def print_report(self) -> None:
        s = self.summary()
        print("=" * 60)
        print("Context Observability Report")
        print("=" * 60)
        print(json.dumps(s, indent=2))
        print("\nCost Attribution:")
        print(json.dumps(self.cost_attribution(), indent=2))


# ---------------------------------------------------------------------------
# OpenTelemetry-compatible span context manager
# ---------------------------------------------------------------------------

class ContextAssemblySpan:
    """
    Lightweight span for timing context assembly stages.
    Compatible with OpenTelemetry span interface — replace with
    opentelemetry.trace.get_tracer(...).start_as_current_span() in production.

    Usage:
        with ContextAssemblySpan("retrieve_rag") as span:
            results = await retrieve_rag(query)
            span.set_attribute("chunks_retrieved", len(results))
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._start: float = 0.0
        self._attributes: dict[str, Any] = {}

    def __enter__(self) -> "ContextAssemblySpan":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        duration_ms = (time.perf_counter() - self._start) * 1000
        self._attributes["duration_ms"] = round(duration_ms, 2)
        logger.debug("Span [%s]: %s", self.name, self._attributes)

    def set_attribute(self, key: str, value: Any) -> None:
        self._attributes[key] = value

    @property
    def duration_ms(self) -> float:
        return self._attributes.get("duration_ms", 0.0)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

def demo() -> None:
    import uuid

    observer = ContextObserver(model="claude-sonnet-4-5", context_window=16_000)
    session_id = "sess_demo_001"

    # Simulate 5 turns of a conversation
    turns = [
        {"system": 500, "rag": 3200, "history": 0,    "query": 120},
        {"system": 500, "rag": 3200, "history": 800,  "query": 95},
        {"system": 500, "rag": 3200, "history": 1600, "query": 140},
        {"system": 500, "rag": 2800, "history": 2400, "query": 110},  # RAG reduced
        {"system": 500, "rag": 2800, "history": 1200, "query": 88},   # History compressed
    ]

    for i, layer_tokens in enumerate(turns, start=1):
        with ContextAssemblySpan("context_assembly") as span:
            span.set_attribute("turn", i)
            span.set_attribute("total_tokens", sum(layer_tokens.values()))

        observer.record(
            request_id=str(uuid.uuid4())[:8],
            session_id=session_id,
            turn=i,
            layer_tokens=layer_tokens,
            cached_tokens=500,  # System prompt always cached
            output_tokens=350,
            compression_applied=(i == 5),
            layers_trimmed=["old_history"] if i == 5 else [],
            assembly_latency_ms=span.duration_ms + 45,  # Add mock retrieval time
        )

    observer.print_report()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
