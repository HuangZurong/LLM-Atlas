"""
Multi-Turn Support Context Manager
===================================
Stateful context management for customer support conversations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..shared.composer import ContextLayer, ContextComposer, count_tokens
from ..shared.budget_controller import TokenBudgetController, BudgetConfig
from ..shared.compressor import AdaptiveCompressor


@dataclass
class SupportState:
    """
    Structured state for customer support conversations.

    This is the "schema-driven state tracking" pattern:
    - Core fields are strongly-typed and code-owned
    - Dynamic fields are LLM-owned (preferences, notes)
    """
    # Core state (code-owned)
    customer_id: str | None = None
    active_order_id: str | None = None
    refund_requested: bool = False
    refund_amount: float | None = None

    # Dynamic state (LLM-owned)
    user_preferences: dict[str, Any] = field(default_factory=dict)
    issue_notes: list[str] = field(default_factory=list)

    def to_context_string(self) -> str:
        """Serialize state to context-friendly string."""
        lines = ["[CURRENT SESSION STATE]"]
        if self.customer_id:
            lines.append(f"Customer ID: {self.customer_id}")
        if self.active_order_id:
            lines.append(f"Active Order: {self.active_order_id}")
        if self.refund_requested:
            lines.append(f"Refund Requested: ${self.refund_amount:.2f}" if self.refund_amount else "Refund Requested: Yes")
        if self.issue_notes:
            lines.append(f"Issue Notes: {'; '.join(self.issue_notes)}")
        return "\n".join(lines)


class SupportContextManager:
    """
    Complete context management for customer support agents.

    Integrates:
    - ContextComposer (priority-based assembly)
    - TokenBudgetController (threshold-based triggers)
    - AdaptiveCompressor (quality-preserving compression)
    - State tracking (order IDs, refund status)
    """

    def __init__(
        self,
        total_budget: int = 16_000,
        output_reserve_ratio: float = 0.20,
    ) -> None:
        self.composer = ContextComposer(total_budget, output_reserve_ratio)
        self.budget_controller = TokenBudgetController(
            BudgetConfig(total_window=total_budget, output_reserve_ratio=output_reserve_ratio)
        )
        self.compressor = AdaptiveCompressor(llm_client=None)
        self.state = SupportState()
        self._turn_count = 0

    def add_turn(self, role: str, content: str) -> None:
        """Add a conversation turn to history."""
        self._turn_count += 1
        # In production, this would update the budget controller
        # and potentially trigger compression

    def update_state(self, **kwargs) -> None:
        """Update the session state."""
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)

    def build_context(
        self,
        system_prompt: str,
        knowledge_base: str,
        history: list[tuple[str, str]],
        user_query: str,
        compress: bool = True,
    ) -> str:
        """
        Build the final context string.

        Returns the assembled context ready for LLM API call.
        """
        # Split history into old and recent
        split_point = max(0, len(history) - 4)
        old_history = "\n".join(f"{r.upper()}: {c}" for r, c in history[:split_point])
        recent_history = "\n".join(f"{r.upper()}: {c}" for r, c in history[split_point:])

        # Add state
        state_text = self.state.to_context_string()

        # Build layers
        layers = [
            ContextLayer("system_prompt", system_prompt, priority="p0"),
            ContextLayer("state", state_text, priority="p1"),
            ContextLayer("user_query", user_query, priority="p1"),
            ContextLayer("recent_history", recent_history, priority="p2"),
            ContextLayer("knowledge_base", knowledge_base, priority="p3"),
            ContextLayer("old_history", old_history, priority="p5"),
        ]

        # Compose
        assembled = self.composer.compose(layers)

        # Report
        print(f"[Context] {assembled.total_tokens}/{assembled.input_budget} tokens ({assembled.utilization:.1%})")
        if assembled.trimmed_layers:
            print(f"[Dropped] {assembled.trimmed_layers}")

        # Build final string
        return "\n\n".join(layer.content for layer in assembled.layers)

    def get_report(self) -> dict:
        """Get utilization report."""
        return {
            "turn_count": self._turn_count,
            "state": {
                "customer_id": self.state.customer_id,
                "active_order_id": self.state.active_order_id,
                "refund_requested": self.state.refund_requested,
            },
        }