"""
Context Composition Pipeline
=============================
Assembles a production-ready context from multiple sources (system prompt,
memory, RAG results, conversation history, user query) while respecting a
token budget and priority-based trimming rules.

Prerequisites:
  - 01_Theory/02_Context_Composition.md
  - 01_Theory/03_Token_Budget_and_Cost.md
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

import tiktoken

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

Priority = Literal["p0", "p1", "p2", "p3", "p4", "p5"]


@dataclass
class ContextLayer:
    name: str
    content: str
    priority: Priority          # p0 = never trim, p5 = trim first
    tokens: int = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = count_tokens(self.content)


@dataclass
class AssembledContext:
    layers: list[ContextLayer]
    total_tokens: int
    budget: int
    trimmed_layers: list[str]   # names of layers that were trimmed/dropped

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


# ---------------------------------------------------------------------------
# Context Composer
# ---------------------------------------------------------------------------

PRIORITY_ORDER: list[Priority] = ["p5", "p4", "p3", "p2", "p1", "p0"]


class ContextComposer:
    """
    Assembles context layers within a token budget.

    Priority-based trimming (trim p5 first, never trim p0):
      P0 — system_prompt      (never trim)
      P1 — user_query         (trim last)
      P2 — recent_history     (keep verbatim)
      P3 — rag_context        (reduce chunks)
      P4 — memory             (compress)
      P5 — old_history        (summarize or drop)
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
        The reminder is injected after composition to ensure it's always present.
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


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

def build_example_context() -> AssembledContext:
    composer = ContextComposer(total_budget=16_000, output_reserve_ratio=0.25)

    layers = [
        ContextLayer(
            name="system_prompt",
            content="You are a helpful customer support assistant for Acme Corp. "
                    "Always be polite, concise, and accurate. "
                    "If you don't know the answer, say so.",
            priority="p0",
        ),
        ContextLayer(
            name="memory",
            content="User [user_id] has contacted support 3 times this month. "
                    "Previous issue: shipping delay on order #1001 (resolved). "
                    "Preferred language: English.",
            priority="p4",
        ),
        ContextLayer(
            name="rag_context",
            content="[FAQ: Refund Policy]\n"
                    "Refunds are processed within 5–7 business days. "
                    "Items must be returned within 30 days of purchase. "
                    "Digital products are non-refundable.",
            priority="p3",
        ),
        ContextLayer(
            name="recent_history",
            content="User: Hi, I need help with my order.\n"
                    "Assistant: Of course! Could you provide your order number?\n"
                    "User: It's order #1002.",
            priority="p2",
        ),
        ContextLayer(
            name="user_query",
            content="I haven't received my order yet and it's been 2 weeks. "
                    "Can I get a refund?",
            priority="p1",
        ),
    ]

    assembled = composer.compose_with_sandwich(
        layers=layers,
        reminder="Remember: always verify the order number before processing any refund.",
    )

    print(f"Context assembled: {assembled.total_tokens}/{assembled.input_budget} tokens "
          f"({assembled.utilization:.1%} utilization)")
    if assembled.trimmed_layers:
        print(f"Trimmed layers: {assembled.trimmed_layers}")

    return assembled


if __name__ == "__main__":
    ctx = build_example_context()
    messages = ctx.to_messages()
    print(f"\nMessages structure:")
    for msg in messages:
        print(f"  [{msg['role']}]: {len(msg['content'])} chars")
