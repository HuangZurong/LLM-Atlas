"""
Token Budget Controller
========================
Tracks token usage across context layers in real time, enforces budget limits,
and triggers compression when thresholds are crossed.

Prerequisites:
  - 01_Theory/03_Token_Budget_and_Cost.md
  - 02_Practical/01_Context_Composition_Pipeline.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import tiktoken

logger = logging.getLogger(__name__)

_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


# ---------------------------------------------------------------------------
# Budget configuration
# ---------------------------------------------------------------------------

@dataclass
class BudgetConfig:
    total_window: int = 128_000
    output_reserve_ratio: float = 0.25

    # Compression trigger thresholds (as fraction of input budget)
    soft_limit: float = 0.70    # Start compressing old history
    hard_limit: float = 0.85    # Aggressive compression
    emergency_limit: float = 0.95  # Drop non-essential content

    # Layer allocations (as fraction of available input budget)
    allocations: dict[str, float] = field(default_factory=lambda: {
        "recent_history": 0.35,
        "rag_context":    0.35,
        "memory":         0.20,
        "old_history":    0.10,
    })

    @property
    def input_budget(self) -> int:
        return int(self.total_window * (1 - self.output_reserve_ratio))


# ---------------------------------------------------------------------------
# Budget controller
# ---------------------------------------------------------------------------

CompressionFn = Callable[[str], str]


class TokenBudgetController:
    """
    Tracks token usage per layer and enforces budget constraints.

    Usage:
        controller = TokenBudgetController(config)
        controller.add("system_prompt", system_text, compressible=False)
        controller.add("rag_context", rag_text, compressible=True)
        controller.add("user_query", query_text, compressible=False)

        if controller.needs_compression():
            controller.compress(summarize_fn)

        final_context = controller.build()
    """

    def __init__(
        self,
        config: BudgetConfig | None = None,
        compress_fn: CompressionFn | None = None,
    ) -> None:
        self.config = config or BudgetConfig()
        self.compress_fn = compress_fn or self._default_truncate
        self._layers: dict[str, dict] = {}  # name → {content, tokens, compressible, priority}

    # ------------------------------------------------------------------
    # Layer management
    # ------------------------------------------------------------------

    def add(
        self,
        name: str,
        content: str,
        compressible: bool = True,
        priority: int = 5,  # 0 = highest (never trim), 5 = lowest (trim first)
    ) -> None:
        self._layers[name] = {
            "content": content,
            "tokens": count_tokens(content),
            "compressible": compressible,
            "priority": priority,
        }

    def update(self, name: str, content: str) -> None:
        if name not in self._layers:
            raise KeyError(f"Layer '{name}' not found. Use add() first.")
        self._layers[name]["content"] = content
        self._layers[name]["tokens"] = count_tokens(content)

    def remove(self, name: str) -> None:
        self._layers.pop(name, None)

    # ------------------------------------------------------------------
    # Budget tracking
    # ------------------------------------------------------------------

    @property
    def total_tokens(self) -> int:
        return sum(l["tokens"] for l in self._layers.values())

    @property
    def utilization(self) -> float:
        return self.total_tokens / self.config.input_budget

    def compression_level(self) -> str:
        u = self.utilization
        if u >= self.config.emergency_limit:
            return "emergency"
        elif u >= self.config.hard_limit:
            return "hard"
        elif u >= self.config.soft_limit:
            return "soft"
        return "none"

    def needs_compression(self) -> bool:
        return self.compression_level() != "none"

    def layer_budget(self, layer_name: str) -> int:
        """Compute the allocated token budget for a named layer."""
        fixed_tokens = sum(
            l["tokens"] for n, l in self._layers.items()
            if n in ("system_prompt", "user_query")
        )
        available = self.config.input_budget - fixed_tokens
        ratio = self.config.allocations.get(layer_name, 0.0)
        return int(available * ratio)

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def compress(self, compress_fn: CompressionFn | None = None) -> list[str]:
        """
        Compress layers starting from lowest priority until under budget.
        Returns list of layer names that were compressed.
        """
        fn = compress_fn or self.compress_fn
        level = self.compression_level()
        compressed: list[str] = []

        # Sort by priority descending (highest priority number = trim first)
        candidates = sorted(
            [(name, layer) for name, layer in self._layers.items()
             if layer["compressible"]],
            key=lambda x: x[1]["priority"],
            reverse=True,
        )

        for name, layer in candidates:
            if not self.needs_compression():
                break

            allocated = self.layer_budget(name)
            if layer["tokens"] > allocated:
                original_tokens = layer["tokens"]
                compressed_content = fn(layer["content"])
                self.update(name, compressed_content)
                compressed.append(name)
                logger.info(
                    "Compressed layer '%s': %d → %d tokens (level: %s)",
                    name, original_tokens, layer["tokens"], level,
                )

        return compressed

    def drop_layer(self, name: str) -> bool:
        """Drop a layer entirely (emergency use). Returns True if dropped."""
        if name in self._layers and self._layers[name]["priority"] >= 4:
            self.remove(name)
            logger.warning("Dropped layer '%s' due to budget emergency.", name)
            return True
        return False

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def build(self) -> str:
        """Build the final context string, ordered by priority (low number first)."""
        ordered = sorted(self._layers.items(), key=lambda x: x[1]["priority"])
        return "\n\n".join(layer["content"] for _, layer in ordered)

    def report(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "input_budget": self.config.input_budget,
            "utilization": f"{self.utilization:.1%}",
            "compression_level": self.compression_level(),
            "layers": {
                name: {
                    "tokens": layer["tokens"],
                    "allocated": self.layer_budget(name),
                    "priority": layer["priority"],
                }
                for name, layer in self._layers.items()
            },
        }

    @staticmethod
    def _default_truncate(text: str, max_chars: int = 2000) -> str:
        """Fallback: simple character truncation with ellipsis."""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n... [truncated]"


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

def demo() -> None:
    config = BudgetConfig(total_window=16_000, output_reserve_ratio=0.25)
    controller = TokenBudgetController(config)

    controller.add("system_prompt",
                   "You are a helpful assistant. Be concise and accurate.",
                   compressible=False, priority=0)
    controller.add("memory",
                   "User prefers formal tone. Past topics: Python, data engineering.",
                   compressible=True, priority=4)
    controller.add("rag_context",
                   "Relevant documentation: " + "context engineering " * 500,
                   compressible=True, priority=3)
    controller.add("recent_history",
                   "User: What is context engineering?\nAssistant: It is...",
                   compressible=True, priority=2)
    controller.add("user_query",
                   "Can you give me a practical example?",
                   compressible=False, priority=1)

    import json
    print("Before compression:")
    print(json.dumps(controller.report(), indent=2))

    if controller.needs_compression():
        compressed = controller.compress()
        print(f"\nCompressed layers: {compressed}")
        print("\nAfter compression:")
        print(json.dumps(controller.report(), indent=2))


if __name__ == "__main__":
    demo()
