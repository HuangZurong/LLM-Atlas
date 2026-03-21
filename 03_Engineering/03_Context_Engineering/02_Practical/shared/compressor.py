"""
Context Compression Strategies
================================
Implements multiple compression strategies for reducing context size while
preserving information quality. Strategies range from simple truncation to
LLM-based abstractive summarization.

Prerequisites:
  - 01_Theory/03_Token_Budget_and_Cost.md
  - 02_Practical/02_Token_Budget_Controller.py
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

import tiktoken

_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class CompressionStrategy(ABC):
    """Base class for all context compression strategies."""

    @abstractmethod
    def compress(self, text: str, target_tokens: int) -> str:
        """Compress text to approximately target_tokens."""
        ...

    def fits(self, text: str, target_tokens: int) -> bool:
        return count_tokens(text) <= target_tokens


# ---------------------------------------------------------------------------
# Strategy 1: Truncation
# ---------------------------------------------------------------------------

class TruncationStrategy(CompressionStrategy):
    """
    Simple truncation — drop tokens from the oldest end.
    Zero cost, high information loss at boundaries.
    Best for: conversation history where recency matters most.
    """

    def __init__(self, keep_end: bool = True) -> None:
        self.keep_end = keep_end  # True = keep recent (end), False = keep beginning

    def compress(self, text: str, target_tokens: int) -> str:
        tokens = _ENCODER.encode(text)
        if len(tokens) <= target_tokens:
            return text

        if self.keep_end:
            kept = tokens[-target_tokens:]
            truncated = _ENCODER.decode(kept)
            return "[... earlier content truncated ...]\n" + truncated
        else:
            kept = tokens[:target_tokens]
            return _ENCODER.decode(kept) + "\n[... later content truncated ...]"


# ---------------------------------------------------------------------------
# Strategy 2: Sliding Window
# ---------------------------------------------------------------------------

class SlidingWindowStrategy(CompressionStrategy):
    """
    Keep the most recent N tokens with optional overlap.
    Best for: multi-turn conversation history.
    """

    def __init__(self, overlap_tokens: int = 128) -> None:
        self.overlap_tokens = overlap_tokens

    def compress(self, text: str, target_tokens: int) -> str:
        tokens = _ENCODER.encode(text)
        if len(tokens) <= target_tokens:
            return text

        # Keep the last target_tokens, with overlap from the previous window
        window_start = max(0, len(tokens) - target_tokens)
        kept = tokens[window_start:]
        result = _ENCODER.decode(kept)
        return f"[Window: last {target_tokens} tokens]\n" + result


# ---------------------------------------------------------------------------
# Strategy 3: Extractive Compression
# ---------------------------------------------------------------------------

class ExtractiveSummaryStrategy(CompressionStrategy):
    """
    Keep the most important sentences based on heuristic scoring.
    Zero LLM cost, moderate information preservation.
    Best for: documents where key sentences are identifiable by structure.
    """

    # Sentences containing these patterns score higher
    IMPORTANCE_PATTERNS = [
        r"\b(important|critical|key|note|warning|must|required|deadline)\b",
        r"\b(result|conclusion|finding|recommendation|decision)\b",
        r"\b(error|fail|issue|problem|bug|exception)\b",
        r"^\s*[-•*]\s",  # Bullet points
        r"^\s*\d+\.",    # Numbered lists
    ]

    def compress(self, text: str, target_tokens: int) -> str:
        if self.fits(text, target_tokens):
            return text

        sentences = self._split_sentences(text)
        scored = [(self._score(s), i, s) for i, s in enumerate(sentences)]
        scored.sort(key=lambda x: (-x[0], x[1]))  # High score first, preserve order

        # Greedily add sentences until we hit the target
        selected: list[tuple[int, str]] = []
        current_tokens = 0

        for score, idx, sentence in scored:
            sentence_tokens = count_tokens(sentence)
            if current_tokens + sentence_tokens <= target_tokens:
                selected.append((idx, sentence))
                current_tokens += sentence_tokens

        # Restore original order
        selected.sort(key=lambda x: x[0])
        return " ".join(s for _, s in selected)

    def _split_sentences(self, text: str) -> list[str]:
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _score(self, sentence: str) -> float:
        score = 0.0
        lower = sentence.lower()
        for pattern in self.IMPORTANCE_PATTERNS:
            if re.search(pattern, lower, re.IGNORECASE):
                score += 1.0
        # Prefer medium-length sentences (not too short, not too long)
        words = len(sentence.split())
        if 10 <= words <= 50:
            score += 0.5
        return score


# ---------------------------------------------------------------------------
# Strategy 4: LLM-based Abstractive Summarization
# ---------------------------------------------------------------------------

class AbstractiveSummaryStrategy(CompressionStrategy):
    """
    Use a cheap LLM to generate a concise summary.
    Higher cost, best information preservation.
    Best for: long-term memory, critical documents.
    """

    SUMMARY_PROMPT = """Summarize the following content concisely, preserving all key facts,
decisions, and action items. Output only the summary, no preamble.

Content:
{content}

Summary (target: ~{target_words} words):"""

    def __init__(self, llm_client: Any, model: str = "gpt-4o-mini") -> None:
        self.llm_client = llm_client
        self.model = model

    def compress(self, text: str, target_tokens: int) -> str:
        if self.fits(text, target_tokens):
            return text

        target_words = int(target_tokens * 0.75)  # ~0.75 tokens per word
        prompt = self.SUMMARY_PROMPT.format(
            content=text,
            target_words=target_words,
        )

        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=target_tokens,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Strategy 5: Entity Compression
# ---------------------------------------------------------------------------

class EntityCompressionStrategy(CompressionStrategy):
    """
    Extract structured facts from text into a compact representation.
    Best for: conversation history where entities and facts matter more than prose.
    """

    EXTRACTION_PROMPT = """Extract the key facts from the following conversation as a
structured list. Each fact should be one line. Include: decisions made,
information provided, user preferences, and action items.

Conversation:
{content}

Key facts:"""

    def __init__(self, llm_client: Any, model: str = "gpt-4o-mini") -> None:
        self.llm_client = llm_client
        self.model = model

    def compress(self, text: str, target_tokens: int) -> str:
        if self.fits(text, target_tokens):
            return text

        prompt = self.EXTRACTION_PROMPT.format(content=text)
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=target_tokens,
        )
        facts = response.choices[0].message.content
        return f"[Extracted facts from conversation history]\n{facts}"


# ---------------------------------------------------------------------------
# Strategy selector
# ---------------------------------------------------------------------------

class AdaptiveCompressor:
    """
    Selects the appropriate compression strategy based on layer type and
    available budget.
    """

    def __init__(self, llm_client: Any | None = None) -> None:
        self.llm_client = llm_client
        self._strategies: dict[str, CompressionStrategy] = {
            "truncation": TruncationStrategy(keep_end=True),
            "sliding_window": SlidingWindowStrategy(),
            "extractive": ExtractiveSummaryStrategy(),
        }
        if llm_client:
            self._strategies["abstractive"] = AbstractiveSummaryStrategy(llm_client)
            self._strategies["entity"] = EntityCompressionStrategy(llm_client)

    def compress_layer(
        self,
        layer_name: str,
        content: str,
        target_tokens: int,
        use_llm: bool = False,
    ) -> str:
        """Choose strategy based on layer type."""
        strategy_map = {
            "old_history":    "entity" if (use_llm and self.llm_client) else "truncation",
            "recent_history": "sliding_window",
            "rag_context":    "extractive",
            "memory":         "abstractive" if (use_llm and self.llm_client) else "extractive",
        }
        strategy_name = strategy_map.get(layer_name, "truncation")
        strategy = self._strategies[strategy_name]
        return strategy.compress(content, target_tokens)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    long_text = """
    The project kickoff meeting was held on March 1st. The team decided to use
    Python for the backend and React for the frontend. John will lead the backend
    development. Sarah is responsible for the UI design. The deadline is April 15th.

    During the technical review, we identified three critical issues: the database
    schema needs normalization, the API rate limiting is not implemented, and the
    authentication flow has a security vulnerability. These must be fixed before launch.

    The budget was approved at $50,000. Marketing will begin in week 3. The product
    manager confirmed that the MVP scope includes user registration, basic search,
    and payment processing. Advanced features are deferred to v2.
    """ * 10  # Make it long

    print(f"Original: {count_tokens(long_text)} tokens")

    # Test extractive compression
    extractive = ExtractiveSummaryStrategy()
    compressed = extractive.compress(long_text, target_tokens=200)
    print(f"Extractive: {count_tokens(compressed)} tokens")
    print(compressed[:300])
