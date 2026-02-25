"""
03_Model_Router.py — Intelligent Model Routing with Fallback

Implements production-grade model routing that:
1. Routes queries to the most appropriate model based on complexity
2. Uses cascading fallback (small → medium → large models)
3. Includes cost-aware routing to optimize expenses
4. Provides health checking and circuit breakers

Patterns:
- Intent classification for routing decisions
- Cascade/fallback chains for reliability
- Cost-aware load balancing
- Real-time model health monitoring
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from pydantic import BaseModel
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Model Definitions
# ---------------------------------------------------------------------------

class ModelTier(str, Enum):
    """Model tiers based on capability and cost."""
    TIER_1_CHEAP = "cheap"      # GPT-4o-mini, Claude Haiku, Qwen-7B
    TIER_2_MID = "mid"         # GPT-4o, Claude Sonnet, Llama-3.1-70B
    TIER_3_EXPENSIVE = "expensive"  # GPT-4o-2024-08, Claude Opus, DeepSeek-V3
    TIER_4_SPECIALIZED = "specialized"  # Code models, vision models


@dataclass
class ModelEndpoint:
    """Configuration for a model inference endpoint."""
    name: str
    tier: ModelTier
    client: Any  # OpenAI/Anthropic/self-hosted client
    cost_per_1k_tokens: float  # USD per 1000 tokens
    max_tokens: int = 4096
    capabilities: List[str] = field(default_factory=list)  # e.g., ["code", "vision", "reasoning"]
    is_healthy: bool = True
    error_count: int = 0
    last_error_time: float = 0.0
    circuit_breaker_threshold: int = 5  # Max errors before tripping
    circuit_breaker_reset_seconds: int = 60  # Reset after 60s


# ---------------------------------------------------------------------------
# Router Class
# ---------------------------------------------------------------------------

class ModelRouter:
    """
    Routes queries to appropriate models with fallback logic.

    Routing logic:
      1. Classify query intent/complexity
      2. Select appropriate model tier
      3. Try primary model, fallback if fails
      4. Track costs and performance
    """

    def __init__(self):
        self.endpoints: Dict[str, ModelEndpoint] = {}
        self.logger = logging.getLogger(__name__)
        self.request_counter = 0

    def register_endpoint(self, endpoint: ModelEndpoint):
        """Register a model endpoint."""
        self.endpoints[endpoint.name] = endpoint

    async def route(
        self,
        query: str,
        streaming: bool = False,
        max_cost_usd: float = 0.10,
        timeout_seconds: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Route a query through the appropriate model chain.
        """
        self.request_counter += 1
        request_id = f"req_{self.request_counter:08d}"

        # Step 1: Analyze query for routing
        intent = await self._classify_intent(query)

        # Step 2: Select model candidates
        candidates = self._select_candidates(intent, max_cost_usd)

        if not candidates:
            return {"error": "No suitable models available", "request_id": request_id}

        # Step 3: Try candidates in order
        result = await self._try_candidates(
            query, candidates, request_id, streaming, timeout_seconds
        )

        # Step 4: Record metrics
        await self._record_metrics(request_id, result, candidates[0] if candidates else None)

        return result

    async def _classify_intent(self, query: str) -> Dict[str, Any]:
        """Classify query intent and complexity."""
        # Simple heuristic-based classification
        # In production, use a small LLM or embedding similarity

        intent = {
            "complexity": "medium",  # low/medium/high
            "domain": "general",     # code/math/writing/analysis
            "requires_specialization": False,
            "estimated_tokens": min(len(query.split()) * 1.5, 500),
        }

        # Heuristic rules
        query_lower = query.lower()

        # Complexity detection
        if len(query) < 50:
            intent["complexity"] = "low"
        elif "step by step" in query_lower or "explain" in query_lower:
            intent["complexity"] = "high"

        # Domain detection
        code_keywords = ["python", "javascript", "function", "def ", "class ", "import "]
        math_keywords = ["calculate", "solve", "equation", "formula"]

        if any(keyword in query_lower for keyword in code_keywords):
            intent["domain"] = "code"
            intent["requires_specialization"] = True
        elif any(keyword in query_lower for keyword in math_keywords):
            intent["domain"] = "math"

        return intent

    def _select_candidates(self, intent: Dict[str, Any], max_cost: float) -> List[ModelEndpoint]:
        """Select model candidates based on intent and cost constraints."""
        candidates = []

        # Filter by health
        healthy_endpoints = [ep for ep in self.endpoints.values() if ep.is_healthy]

        # Filter by capability if specialized
        if intent["requires_specialization"]:
            capable_endpoints = [
                ep for ep in healthy_endpoints
                if intent["domain"] in ep.capabilities
            ]
            if capable_endpoints:
                healthy_endpoints = capable_endpoints

        # Group by tier
        tier_mapping = {
            "low": ModelTier.TIER_1_CHEAP,
            "medium": ModelTier.TIER_2_MID,
            "high": ModelTier.TIER_3_EXPENSIVE,
        }

        target_tier = tier_mapping.get(intent["complexity"], ModelTier.TIER_2_MID)

        # Cost-aware selection
        for endpoint in healthy_endpoints:
            # Estimate cost
            estimated_tokens = intent["estimated_tokens"]
            estimated_cost = (estimated_tokens / 1000) * endpoint.cost_per_1k_tokens

            if estimated_cost <= max_cost:
                # Score by tier match (lower is better)
                tier_score = self._tier_distance(endpoint.tier, target_tier)
                candidates.append((tier_score, estimated_cost, endpoint))

        # Sort by tier match, then cost
        candidates.sort(key=lambda x: (x[0], x[1]))
        return [c[2] for c in candidates]

    def _tier_distance(self, tier_a: ModelTier, tier_b: ModelTier) -> int:
        """Calculate distance between tiers."""
        tier_order = [
            ModelTier.TIER_1_CHEAP,
            ModelTier.TIER_2_MID,
            ModelTier.TIER_3_EXPENSIVE,
            ModelTier.TIER_4_SPECIALIZED,
        ]
        idx_a = tier_order.index(tier_a) if tier_a in tier_order else 0
        idx_b = tier_order.index(tier_b) if tier_b in tier_order else 0
        return abs(idx_a - idx_b)

    async def _try_candidates(
        self,
        query: str,
        candidates: List[ModelEndpoint],
        request_id: str,
        streaming: bool,
        timeout: float,
    ) -> Dict[str, Any]:
        """Try candidates in order until success."""
        errors = []

        for endpoint in candidates:
            try:
                self.logger.info(f"Trying {endpoint.name} for request {request_id}")

                # Set timeout
                async with asyncio.timeout(timeout):
                    # Call the model (simplified - would integrate with actual client)
                    result = await self._call_model(endpoint, query, streaming)

                    # Success - reset error counter
                    endpoint.error_count = 0
                    endpoint.is_healthy = True

                    return {
                        "text": result,
                        "model": endpoint.name,
                        "tier": endpoint.tier.value,
                        "request_id": request_id,
                        "fallback_used": endpoint != candidates[0],
                    }

            except Exception as e:
                self.logger.warning(f"Model {endpoint.name} failed: {e}")
                errors.append(f"{endpoint.name}: {str(e)}")

                # Update health status
                endpoint.error_count += 1
                endpoint.last_error_time = time.time()

                if endpoint.error_count >= endpoint.circuit_breaker_threshold:
                    endpoint.is_healthy = False
                    self.logger.error(f"Circuit breaker tripped for {endpoint.name}")

        # All candidates failed
        return {
            "error": "All model endpoints failed",
            "errors": errors,
            "request_id": request_id,
        }

    async def _call_model(
        self,
        endpoint: ModelEndpoint,
        query: str,
        streaming: bool = False
    ) -> str:
        """Call a model endpoint."""
        # Simplified - in reality would integrate with OpenAI/Anthropic/vLLM
        # This is a mock implementation

        # Simulate API call
        await asyncio.sleep(0.1)

        # Mock response based on model tier
        if endpoint.tier == ModelTier.TIER_1_CHEAP:
            return f"[Cheap model response to: {query[:50]}...]"
        elif endpoint.tier == ModelTier.TIER_2_MID:
            return f"[Mid-tier model provides detailed analysis of: {query[:50]}...]"
        elif endpoint.tier == ModelTier.TIER_3_EXPENSIVE:
            return f"[Expensive model gives comprehensive answer with reasoning for: {query[:50]}...]"
        else:
            return f"[Specialized model handles: {query[:50]}...]"

    async def _record_metrics(
        self,
        request_id: str,
        result: Dict[str, Any],
        primary_endpoint: Optional[ModelEndpoint]
    ):
        """Record routing metrics."""
        # In production, send to Prometheus/Datadog
        success = "error" not in result
        model_used = result.get("model", "unknown")
        fallback_used = result.get("fallback_used", False)

        self.logger.info(
            f"Request {request_id}: success={success}, model={model_used}, "
            f"fallback={fallback_used}"
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check health of all endpoints."""
        healthy = []
        unhealthy = []

        for name, endpoint in self.endpoints.items():
            # Reset circuit breaker if enough time has passed
            if (not endpoint.is_healthy and
                time.time() - endpoint.last_error_time > endpoint.circuit_breaker_reset_seconds):
                endpoint.is_healthy = True
                endpoint.error_count = 0
                self.logger.info(f"Circuit breaker reset for {name}")

            if endpoint.is_healthy:
                healthy.append(name)
            else:
                unhealthy.append(name)

        return {
            "total_endpoints": len(self.endpoints),
            "healthy": healthy,
            "unhealthy": unhealthy,
            "healthy_ratio": len(healthy) / len(self.endpoints) if self.endpoints else 0,
        }


# ---------------------------------------------------------------------------
# Example Setup
# ---------------------------------------------------------------------------

async def main():
    """Example of model routing."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    router = ModelRouter()

    # Register mock endpoints (in production, use real clients)
    router.register_endpoint(ModelEndpoint(
        name="gpt-4o-mini",
        tier=ModelTier.TIER_1_CHEAP,
        client=None,
        cost_per_1k_tokens=0.15,
        capabilities=["general", "writing"],
    ))

    router.register_endpoint(ModelEndpoint(
        name="gpt-4o",
        tier=ModelTier.TIER_2_MID,
        client=None,
        cost_per_1k_tokens=2.50,
        capabilities=["general", "reasoning", "analysis"],
    ))

    router.register_endpoint(ModelEndpoint(
        name="claude-3-5-sonnet",
        tier=ModelTier.TIER_3_EXPENSIVE,
        client=None,
        cost_per_1k_tokens=3.00,
        capabilities=["general", "reasoning", "analysis", "writing"],
    ))

    router.register_endpoint(ModelEndpoint(
        name="deepseek-coder",
        tier=ModelTier.TIER_4_SPECIALIZED,
        client=None,
        cost_per_1k_tokens=0.80,
        capabilities=["code"],
        max_tokens=16384,
    ))

    # Test queries
    test_queries = [
        "Hello, how are you?",  # Simple → cheap tier
        "Explain quantum computing in simple terms",  # Medium complexity → mid tier
        "Write a Python function to find all prime numbers up to N",  # Code → specialized
        "Solve this complex physics problem: F=ma and explain step by step",  # Complex → expensive
    ]

    print("Testing model routing...")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQuery: {query[:80]}...")
        result = await router.route(query, max_cost_usd=0.50)

        if "error" in result:
            print(f"  ❌ Error: {result['error']}")
        else:
            print(f"  ✅ Model: {result['model']} (Tier: {result['tier']})")
            print(f"  Text: {result['text'][:100]}...")
            if result.get('fallback_used'):
                print(f"  ⚠️  Used fallback model")

    # Health check
    print("\n" + "=" * 60)
    print("Health check:")
    health = await router.health_check()
    for key, value in health.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())