"""
Context Composer
================
Assembles context from multiple sources while respecting token budget and priority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import tiktoken

_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken (cl100k_base encoding)."""
    return len(_ENCODER.encode(text))


Priority = Literal["p0", "p1", "p2", "p3", "p4", "p5"]


@dataclass
class ContextLayer:
    """
    A single layer of context with priority.

    Priority levels:
      P0 — never trim (e.g., system_prompt)
      P1 — trim last (e.g., user_query)
      P2 — keep verbatim (e.g., recent_history)
      P3 — reduce (e.g., rag_context)
      P4 — compress (e.g., memory)
      P5 — trim first (e.g., old_history)
    """
    name: str
    content: str
    priority: Priority
    tokens: int = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = count_tokens(self.content)


@dataclass
class AssembledContext:
    """Result of context assembly with token tracking."""
    layers: list[ContextLayer]
    total_tokens: int
    budget: int
    trimmed_layers: list[str]

    @property
    def utilization(self) -> float:
        return self.total_tokens / self.budget

    def to_messages(self) -> list[dict]:
        """Convert to OpenAI-compatible messages format."""
        system_layers = [l for l in self.layers if l.name == "system_prompt"]
        other_layers = [l for l in self.layers if l.name != "system_prompt"]

        system_content = "\n\n".join(l.content for l in system_layers)
        user_content = "\n\n".join(
            f"[{l.name.upper()}]\n{l.content}" for l in other_layers
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]


PRIORITY_ORDER: list[Priority] = ["p5", "p4", "p3", "p2", "p1", "p0"]


class ContextComposer:
    """
    Assembles context layers within a token budget.

    Trims lowest-priority layers first when over budget.
    """

    def __init__(
        self,
        total_budget: int = 16_000,
        output_reserve_ratio: float = 0.25,
    ) -> None:
        self.total_budget = total_budget
        self.input_budget = int(total_budget * (1 - output_reserve_ratio))

    def compose(self, layers: list[ContextLayer]) -> AssembledContext:
        """
        Assemble layers into a context that fits within the input budget.
        Trims lower-priority layers first when over budget.
        """
        trimmed: list[str] = []

        # Sort: highest priority (p0) last so we trim from the front
        sorted_layers = sorted(layers, key=lambda l: l.priority, reverse=True)

        total = sum(l.tokens for l in sorted_layers)

        # Trim until we fit
        for priority in PRIORITY_ORDER:
            if total <= self.input_budget:
                break
            candidates = [l for l in sorted_layers if l.priority == priority]
            for layer in candidates:
                if total <= self.input_budget:
                    break
                sorted_layers.remove(layer)
                total -= layer.tokens
                trimmed.append(layer.name)

        # Restore original order (static first, dynamic last for prefix caching)
        priority_rank = {"p0": 0, "p1": 1, "p2": 2, "p3": 3, "p4": 4, "p5": 5}
        final_layers = sorted(sorted_layers, key=lambda l: priority_rank[l.priority])

        return AssembledContext(
            layers=final_layers,
            total_tokens=total,
            budget=self.input_budget,
            trimmed_layers=trimmed,
        )

    def compose_with_sandwich(self, layers: list[ContextLayer], reminder: str) -> AssembledContext:
        """
        Apply the Sandwich Pattern: append a key-constraint reminder at the end.
        """
        assembled = self.compose(layers)
        reminder_layer = ContextLayer(
            name="reminder",
            content=reminder,
            priority="p0",
        )
        assembled.layers.append(reminder_layer)
        assembled.total_tokens += reminder_layer.tokens
        return assembled