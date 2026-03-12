"""
Multi-Turn Context Manager
===========================
Manages context state across a multi-turn conversation, handling history
accumulation, compression triggers, and context assembly for each turn.

Integrates: TokenBudgetController + ContextComposer + AdaptiveCompressor

Prerequisites:
  - 01_Theory/05_Dynamic_Context_Management.md
  - 02_Practical/01_Context_Composition_Pipeline.py
  - 02_Practical/02_Token_Budget_Controller.py
  - 02_Practical/03_Context_Compression.py
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)

_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    role: str   # "user" or "assistant"
    content: str
    tokens: int = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = count_tokens(self.content)

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class ContextSnapshot:
    """Represents the assembled context for a single LLM call."""
    system_prompt: str
    history: list[Turn]
    rag_context: str
    memory: str
    user_query: str
    total_tokens: int
    compression_applied: bool
    turn_number: int

    def to_messages(self) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]

        if self.memory:
            messages.append({
                "role": "system",
                "content": f"[RELEVANT MEMORY]\n{self.memory}",
            })
        if self.rag_context:
            messages.append({
                "role": "system",
                "content": f"[RETRIEVED CONTEXT]\n{self.rag_context}",
            })

        messages.extend(t.to_dict() for t in self.history)
        messages.append({"role": "user", "content": self.user_query})
        return messages


# ---------------------------------------------------------------------------
# Multi-turn context manager
# ---------------------------------------------------------------------------

class MultiTurnContextManager:
    """
    Manages context across a multi-turn conversation session.

    Responsibilities:
    - Accumulate conversation history
    - Trigger compression when approaching budget limits
    - Assemble context for each LLM call
    - Track context metrics per turn
    """

    # Compression thresholds (fraction of input budget)
    SOFT_LIMIT = 0.70
    HARD_LIMIT = 0.85

    # Recent history to always keep verbatim (number of turns)
    RECENT_TURNS_VERBATIM = 4

    def __init__(
        self,
        system_prompt: str,
        total_budget: int = 16_000,
        output_reserve_ratio: float = 0.25,
        llm_client: Any | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.total_budget = total_budget
        self.input_budget = int(total_budget * (1 - output_reserve_ratio))
        self.llm_client = llm_client

        self._history: list[Turn] = []
        self._compressed_summary: str = ""  # Summary of old turns
        self._turn_number: int = 0
        self._metrics: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assemble(
        self,
        user_query: str,
        rag_context: str = "",
        memory: str = "",
    ) -> ContextSnapshot:
        """
        Assemble the context for the next LLM call.
        Call this before each LLM invocation.
        """
        self._turn_number += 1
        compression_applied = False

        # Check if we need to compress before assembling
        estimated = self._estimate_tokens(user_query, rag_context, memory)
        if estimated > self.input_budget * self.HARD_LIMIT:
            self._compress_history()
            compression_applied = True
        elif estimated > self.input_budget * self.SOFT_LIMIT:
            self._trim_old_history()
            compression_applied = True

        # Build history to include
        history_to_include = self._build_history_for_context()

        # Assemble final context
        total = (
            count_tokens(self.system_prompt)
            + count_tokens(memory)
            + count_tokens(rag_context)
            + sum(t.tokens for t in history_to_include)
            + count_tokens(user_query)
        )

        snapshot = ContextSnapshot(
            system_prompt=self.system_prompt,
            history=history_to_include,
            rag_context=rag_context,
            memory=memory,
            user_query=user_query,
            total_tokens=total,
            compression_applied=compression_applied,
            turn_number=self._turn_number,
        )

        self._record_metric(snapshot)
        return snapshot

    def record_response(self, assistant_response: str) -> None:
        """
        Record the assistant's response after the LLM call.
        Call this after each LLM invocation.
        """
        # Record the last user query (we need to find it from the last assemble call)
        # In practice, you'd pass the query here too, but we keep it simple
        self._history.append(Turn(role="assistant", content=assistant_response))

    def add_user_turn(self, content: str) -> None:
        """Add a user turn to history."""
        self._history.append(Turn(role="user", content=content))

    def get_metrics(self) -> list[dict]:
        return self._metrics

    def export_state(self) -> dict:
        """Export session state for checkpointing."""
        return {
            "system_prompt": self.system_prompt,
            "compressed_summary": self._compressed_summary,
            "recent_history": [t.to_dict() for t in self._history[-self.RECENT_TURNS_VERBATIM * 2:]],
            "turn_number": self._turn_number,
            "metrics": self._metrics,
        }

    @classmethod
    def restore_from_checkpoint(
        cls,
        state: dict,
        total_budget: int = 16_000,
        llm_client: Any | None = None,
    ) -> "MultiTurnContextManager":
        """Restore a session from a checkpoint."""
        manager = cls(
            system_prompt=state["system_prompt"],
            total_budget=total_budget,
            llm_client=llm_client,
        )
        manager._compressed_summary = state.get("compressed_summary", "")
        manager._turn_number = state.get("turn_number", 0)
        for turn_dict in state.get("recent_history", []):
            manager._history.append(Turn(**turn_dict))
        return manager

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_tokens(self, query: str, rag: str, memory: str) -> int:
        history_tokens = sum(t.tokens for t in self._history)
        return (
            count_tokens(self.system_prompt)
            + count_tokens(memory)
            + count_tokens(rag)
            + history_tokens
            + count_tokens(query)
        )

    def _build_history_for_context(self) -> list[Turn]:
        """
        Build the history list to include in context.
        Always includes recent turns verbatim; older turns as summary.
        """
        recent_cutoff = self.RECENT_TURNS_VERBATIM * 2  # user + assistant pairs
        recent = self._history[-recent_cutoff:] if len(self._history) > recent_cutoff else self._history

        if self._compressed_summary:
            summary_turn = Turn(
                role="system",
                content=f"[Earlier conversation summary]\n{self._compressed_summary}",
            )
            return [summary_turn] + list(recent)
        return list(recent)

    def _trim_old_history(self) -> None:
        """Soft compression: drop oldest turns beyond the recent window."""
        recent_cutoff = self.RECENT_TURNS_VERBATIM * 2
        if len(self._history) > recent_cutoff:
            dropped = self._history[:-recent_cutoff]
            self._history = self._history[-recent_cutoff:]
            logger.info("Soft trim: dropped %d old turns", len(dropped))

    def _compress_history(self) -> None:
        """Hard compression: summarize old history using LLM or extractive fallback."""
        recent_cutoff = self.RECENT_TURNS_VERBATIM * 2
        if len(self._history) <= recent_cutoff:
            return

        old_turns = self._history[:-recent_cutoff]
        self._history = self._history[-recent_cutoff:]

        old_text = "\n".join(
            f"{t.role.capitalize()}: {t.content}" for t in old_turns
        )

        if self.llm_client:
            summary = self._llm_summarize(old_text)
        else:
            # Fallback: keep first and last sentences of each turn
            summary = self._extractive_summarize(old_text)

        # Append to existing summary if there is one
        if self._compressed_summary:
            self._compressed_summary += f"\n\n{summary}"
        else:
            self._compressed_summary = summary

        logger.info("Compressed %d old turns into summary (%d tokens)",
                    len(old_turns), count_tokens(self._compressed_summary))

    def _llm_summarize(self, text: str) -> str:
        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Summarize this conversation history concisely, "
                           f"preserving key facts and decisions:\n\n{text}",
            }],
            max_tokens=500,
        )
        return response.choices[0].message.content

    @staticmethod
    def _extractive_summarize(text: str) -> str:
        """Simple extractive fallback: keep first sentence of each turn."""
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        summary_lines = []
        for line in lines:
            first_sentence = line.split(".")[0] + "."
            summary_lines.append(first_sentence)
        return "[Summary] " + " ".join(summary_lines[:10])

    def _record_metric(self, snapshot: ContextSnapshot) -> None:
        self._metrics.append({
            "turn": snapshot.turn_number,
            "total_tokens": snapshot.total_tokens,
            "utilization": f"{snapshot.total_tokens / self.input_budget:.1%}",
            "compression_applied": snapshot.compression_applied,
            "history_turns": len(snapshot.history),
        })


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

def demo() -> None:
    manager = MultiTurnContextManager(
        system_prompt="You are a helpful assistant.",
        total_budget=4_000,  # Small budget to trigger compression quickly
    )

    # Simulate a multi-turn conversation
    turns = [
        ("What is context engineering?", "It's the practice of managing what goes into the LLM context window."),
        ("Can you give an example?", "Sure — deciding whether to include full conversation history or a summary."),
        ("What about token budgets?", "Token budgets allocate space across system prompt, memory, RAG, and history."),
        ("How do I handle long conversations?", "Use compression: summarize old turns, keep recent ones verbatim."),
        ("What compression strategies exist?", "Truncation, sliding window, extractive summary, LLM-based summary."),
    ]

    for user_msg, assistant_msg in turns:
        manager.add_user_turn(user_msg)
        snapshot = manager.assemble(user_query=user_msg)
        print(f"Turn {snapshot.turn_number}: {snapshot.total_tokens} tokens "
              f"({'compressed' if snapshot.compression_applied else 'ok'})")
        manager.record_response(assistant_msg)

    print("\nMetrics:")
    print(json.dumps(manager.get_metrics(), indent=2))


if __name__ == "__main__":
    demo()
