"""
Long Document Processor
========================
Context engineering for documents that exceed the context window.

Implements:
  - Three chunking strategies (fixed-size, semantic, sliding-window)
  - Map-Reduce for parallel chunk processing
  - Hierarchical summarization (tree-of-summaries)
  - Position-aware assembly (Sandwich Pattern)
  - Integration with shared/ budget control and observability

Prerequisites:
  - 01_Theory/04_Long_Context_Techniques.md
  - 02_Practical/shared/
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from ..shared.composer import ContextComposer, count_tokens
from ..shared.compressor import ExtractiveSummaryStrategy


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    """A single chunk of a document with token count and position metadata."""
    index: int
    content: str
    tokens: int = field(init=False)
    strategy: str = "unknown"

    def __post_init__(self) -> None:
        self.tokens = count_tokens(self.content)


@dataclass
class ProcessingResult:
    """Output of DocumentAnalysisProcessor.process()."""
    assembled_context: str
    total_tokens: int
    budget: int
    chunks_produced: int
    tree_depth: int            # 0 if hierarchical summarization was not used
    compression_applied: bool
    trimmed_layers: list[str]

    @property
    def utilization(self) -> float:
        return self.total_tokens / self.budget if self.budget else 0.0

    def summary(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "budget": self.budget,
            "utilization": f"{self.utilization:.1%}",
            "chunks_produced": self.chunks_produced,
            "tree_depth": self.tree_depth,
            "compression_applied": self.compression_applied,
            "trimmed_layers": self.trimmed_layers,
        }


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------

def fixed_size_chunk(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Split text into fixed-size token chunks with optional overlap.

    Predictable and fast. Breaks at word boundaries, not sentence boundaries.
    Use when document structure is irrelevant (e.g., raw logs, code).

    Args:
        text:       Input text.
        chunk_size: Maximum tokens per chunk.
        overlap:    Tokens to repeat at the start of each new chunk.

    Returns:
        List of text chunks.
    """
    words = text.split()
    # Approximate: 1 word ≈ 1.3 tokens
    words_per_chunk = max(1, int(chunk_size / 1.3))
    words_overlap = max(0, int(overlap / 1.3))
    stride = words_per_chunk - words_overlap

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + words_per_chunk, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += stride
    return chunks


def semantic_chunk(text: str, max_tokens: int = 512) -> list[str]:
    """
    Split on paragraph boundaries (double newline), merging small paragraphs
    and splitting oversized ones at sentence boundaries.

    Preserves semantic coherence. Preferred for prose documents.

    Args:
        text:       Input text.
        max_tokens: Token budget per chunk.

    Returns:
        List of semantically coherent chunks.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if para_tokens > max_tokens:
            # Flush current buffer first
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts, current_tokens = [], 0
            # Split the oversized paragraph at sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", para)
            sub_parts: list[str] = []
            sub_tokens = 0
            for sent in sentences:
                st = count_tokens(sent)
                if sub_tokens + st > max_tokens and sub_parts:
                    chunks.append(" ".join(sub_parts))
                    sub_parts, sub_tokens = [], 0
                sub_parts.append(sent)
                sub_tokens += st
            if sub_parts:
                chunks.append(" ".join(sub_parts))

        elif current_tokens + para_tokens > max_tokens:
            if current_parts:
                chunks.append("\n\n".join(current_parts))
            current_parts = [para]
            current_tokens = para_tokens
        else:
            current_parts.append(para)
            current_tokens += para_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def sliding_window_chunk(text: str, window_size: int = 512, stride: int = 384) -> list[str]:
    """
    Sliding window chunking with configurable stride and overlap.

    Each chunk overlaps the previous by (window_size - stride) tokens.
    Use when context continuity across chunk boundaries is critical (e.g., retrieval).

    Args:
        text:        Input text.
        window_size: Tokens per window.
        stride:      Tokens to advance between windows.

    Returns:
        List of overlapping text chunks.
    """
    words = text.split()
    words_per_window = max(1, int(window_size / 1.3))
    words_stride = max(1, int(stride / 1.3))

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + words_per_window, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += words_stride
    return chunks


CHUNKING_STRATEGIES: dict[str, Callable] = {
    "fixed":    fixed_size_chunk,
    "semantic": semantic_chunk,
    "sliding":  sliding_window_chunk,
}
# Used only for documentation/introspection; chunk() dispatches explicitly.


# ---------------------------------------------------------------------------
# Map-Reduce
# ---------------------------------------------------------------------------

class MapReduceProcessor:
    """
    Generic map-reduce processor for long documents.

    Splits the document into chunks, applies a map function to each chunk
    independently, then aggregates the results with a reduce function.

    When to use:
      - Summarizing long reports
      - Extracting structured data from large documents
      - Any task where chunk results can be combined independently

    Args:
        map_fn:    Applied to each chunk. Returns a string result.
        reduce_fn: Aggregates all map results into a single string.
    """

    def __init__(
        self,
        map_fn: Callable[[str], str],
        reduce_fn: Callable[[list[str]], str],
    ) -> None:
        self.map_fn = map_fn
        self.reduce_fn = reduce_fn

    def process(self, text: str, chunk_size: int = 512) -> str:
        """
        Split → map each chunk → reduce all results.

        Args:
            text:       Full document text.
            chunk_size: Token budget per chunk (uses fixed_size_chunk).

        Returns:
            Final aggregated result string.
        """
        chunks = fixed_size_chunk(text, chunk_size=chunk_size, overlap=0)
        mapped = [self.map_fn(chunk) for chunk in chunks]
        return self.reduce_fn(mapped)


# ---------------------------------------------------------------------------
# Hierarchical Summarization
# ---------------------------------------------------------------------------

class HierarchicalSummarizer:
    """
    Build a summary tree over a list of text chunks (tree-of-summaries).

    At each level, groups of `branch_factor` chunks are summarized together.
    Recursively summarizes until a single root summary remains.

    When to use:
      - Documents too large for a single map-reduce pass
      - Need a multi-level summary (section → chapter → document)
      - Want to expose intermediate summaries for different granularities

    Args:
        summarize_fn:  Applied to a combined group of chunks. Returns a summary.
        branch_factor: How many chunks to merge per level (default 4).
    """

    def __init__(
        self,
        summarize_fn: Callable[[str], str],
        branch_factor: int = 4,
    ) -> None:
        self.summarize_fn = summarize_fn
        self.branch_factor = branch_factor

    def build_tree(self, chunks: list[str]) -> dict:
        """
        Recursively build the summary tree.

        Returns:
            dict with keys:
              'levels': list of lists (level 0 = original chunks)
              'root':   final single summary string
        """
        levels = [chunks[:]]
        current = chunks[:]

        while len(current) > 1:
            next_level: list[str] = []
            for i in range(0, len(current), self.branch_factor):
                group = current[i: i + self.branch_factor]
                combined = "\n\n".join(group)
                next_level.append(self.summarize_fn(combined))
            levels.append(next_level)
            current = next_level

        return {"levels": levels, "root": current[0]}


# ---------------------------------------------------------------------------
# Position-Aware Assembly
# ---------------------------------------------------------------------------

def position_aware_assemble(
    instruction: str,
    background: str,
    relevant_chunks: list[str],
    query: str,
    composer: ContextComposer | None = None,
) -> tuple[str, list[str]]:
    """
    Assemble a context using the Sandwich Pattern.

    Builds the context string directly in the correct positional order,
    exploiting the U-shaped attention effect (start and end receive more
    attention than the middle).

    Layout:
      1. instruction       (start — high attention)
      2. background        (static, prefix-cacheable)
      3. chunks[1:]        (less-relevant — buried in the middle)
      4. chunks[0]         (most-relevant — just before query)
      5. query             (near end — high attention)
      6. instruction copy  (Sandwich reminder — end)

    Note: the `composer` argument is accepted for API consistency but
    positional ordering is handled explicitly here, not by composer's
    priority-rank sort (which is designed for trimming, not placement).

    Args:
        instruction:     Task instruction / system prompt.
        background:      Static background knowledge (cacheable).
        relevant_chunks: Retrieved chunks, ordered most-relevant first.
        query:           The user's question.
        composer:        Unused; reserved for future budget-check integration.

    Returns:
        Tuple of (assembled_context: str, trimmed_layers: list[str]).
    """
    SEP = "\n" + "-" * 60 + "\n"
    parts: list[str] = [f"[INSTRUCTION]\n{instruction}"]

    if background:
        parts.append(f"[BACKGROUND]\n{background}")

    # Less-relevant chunks in the middle (indices 1+)
    for i, chunk in enumerate(relevant_chunks[1:], start=2):
        parts.append(f"[CONTEXT CHUNK {i}]\n{chunk}")

    # Most-relevant chunk just before the query
    if relevant_chunks:
        parts.append(f"[MOST RELEVANT CHUNK]\n{relevant_chunks[0]}")

    parts.append(f"[QUERY]\n{query}")
    parts.append(f"[REMINDER]\n{instruction}")

    return SEP.join(parts), []


# ---------------------------------------------------------------------------
# Top-Level Processor
# ---------------------------------------------------------------------------

class DocumentAnalysisProcessor:
    """
    Complete context engineering pipeline for long document analysis.

    Integrates:
    - Chunking strategies (fixed, semantic, sliding-window)
    - Map-Reduce processing
    - Hierarchical summarization
    - Position-aware assembly via ContextComposer (Sandwich Pattern)
    - Extractive compression as default summarize_fn

    Usage:
        processor = DocumentAnalysisProcessor(total_budget=32_000)

        result = processor.process(
            document=long_text,
            query="What are the key findings?",
            instruction="Answer using only the provided context.",
        )
        print(result.summary())
    """

    def __init__(
        self,
        total_budget: int = 128_000,
        output_reserve_ratio: float = 0.25,
        chunk_size: int = 512,
    ) -> None:
        self.total_budget = total_budget
        self.chunk_size = chunk_size
        self.composer = ContextComposer(total_budget, output_reserve_ratio)
        self.compressor = ExtractiveSummaryStrategy()

    # ------------------------------------------------------------------
    # Step 1: Chunk
    # ------------------------------------------------------------------

    def chunk(
        self,
        text: str,
        strategy: str = "semantic",
        chunk_size: int | None = None,
    ) -> list[DocumentChunk]:
        """
        Split a document into chunks.

        Args:
            text:       Document text.
            strategy:   "fixed" | "semantic" | "sliding"
            chunk_size: Override default chunk_size.

        Returns:
            List of DocumentChunk objects.
        """
        size = chunk_size or self.chunk_size
        if strategy == "fixed":
            raw_chunks = fixed_size_chunk(text, chunk_size=size)
        elif strategy == "sliding":
            raw_chunks = sliding_window_chunk(text, window_size=size)
        else:
            raw_chunks = semantic_chunk(text, max_tokens=size)
        return [
            DocumentChunk(index=i, content=c, strategy=strategy)
            for i, c in enumerate(raw_chunks)
        ]

    # ------------------------------------------------------------------
    # Step 2: Map-Reduce (optional)
    # ------------------------------------------------------------------

    def map_reduce(
        self,
        text: str,
        map_fn: Callable[[str], str],
        reduce_fn: Callable[[list[str]], str],
    ) -> str:
        """Run a map-reduce pass over the document."""
        processor = MapReduceProcessor(map_fn, reduce_fn)
        return processor.process(text, chunk_size=self.chunk_size)

    # ------------------------------------------------------------------
    # Step 3: Hierarchical summarization (optional)
    # ------------------------------------------------------------------

    def hierarchical_summarize(
        self,
        chunks: list[str],
        summarize_fn: Callable[[str], str],
        branch_factor: int = 4,
    ) -> dict:
        """
        Build a summary tree over chunks.

        Returns dict with 'levels' and 'root'.
        """
        summarizer = HierarchicalSummarizer(summarize_fn, branch_factor)
        return summarizer.build_tree(chunks)

    # ------------------------------------------------------------------
    # Step 4: Assemble final context
    # ------------------------------------------------------------------

    def assemble(
        self,
        instruction: str,
        background: str,
        relevant_chunks: list[str],
        query: str,
    ) -> tuple[str, list[str]]:
        """Position-aware context assembly using the Sandwich Pattern."""
        return position_aware_assemble(
            instruction=instruction,
            background=background,
            relevant_chunks=relevant_chunks,
            query=query,
            composer=self.composer,
        )

    # ------------------------------------------------------------------
    # Top-level convenience
    # ------------------------------------------------------------------

    def process(
        self,
        document: str,
        query: str,
        instruction: str,
        chunk_strategy: str = "semantic",
        use_hierarchical: bool = True,
        summarize_fn: Callable[[str], str] | None = None,
        top_k_chunks: int = 5,
    ) -> ProcessingResult:
        """
        Full pipeline: chunk → (optional) hierarchical summarize → assemble.

        Pipeline:
          1. Chunk the document using `chunk_strategy`
          2. If `use_hierarchical`, build a summary tree and use root as background
          3. Select `top_k_chunks` as relevant context
          4. Assemble with position-aware Sandwich Pattern

        Args:
            document:          Full document text.
            query:             The user question.
            instruction:       Task instruction placed at start and end.
            chunk_strategy:    "fixed" | "semantic" | "sliding"
            use_hierarchical:  Whether to build a summary tree for background.
            summarize_fn:      Summarization function for hierarchical pass.
                               Defaults to extractive compression if None.
            top_k_chunks:      Number of chunks to include as relevant context.

        Returns:
            ProcessingResult with assembled context and telemetry.
        """
        # Default summarize_fn: extractive compression to 200 tokens
        _summarize_fn = summarize_fn or (
            lambda text: self.compressor.compress(text, target_tokens=200)
        )

        # Step 1: chunk
        doc_chunks = self.chunk(document, strategy=chunk_strategy)
        raw_chunks = [c.content for c in doc_chunks]

        # Step 2: hierarchical summarization for background
        tree_depth = 0
        background = ""
        if use_hierarchical and len(raw_chunks) > 1:
            tree = self.hierarchical_summarize(raw_chunks, _summarize_fn)
            background = tree["root"]
            tree_depth = len(tree["levels"])
        elif raw_chunks:
            # Fallback: use first chunk as background
            background = raw_chunks[0]

        # Step 3: select top-k relevant chunks (simple heuristic: first K)
        # In production, replace with embedding-based similarity ranking
        relevant = raw_chunks[:top_k_chunks]

        # Step 4: assemble
        context_str, trimmed = self.assemble(instruction, background, relevant, query)
        total_tokens = count_tokens(context_str)

        return ProcessingResult(
            assembled_context=context_str,
            total_tokens=total_tokens,
            budget=self.composer.input_budget,
            chunks_produced=len(doc_chunks),
            tree_depth=tree_depth,
            compression_applied=bool(trimmed),
            trimmed_layers=trimmed,
        )
