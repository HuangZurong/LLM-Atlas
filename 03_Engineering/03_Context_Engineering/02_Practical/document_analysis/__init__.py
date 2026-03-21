"""
Document Analysis Scenario
==========================
Context engineering for long document processing.

Key challenges:
- Documents that exceed context window limits
- Need to find relevant sections for specific queries
- Balancing granularity vs. token budget
- Preserving document structure
"""

from .document_processor import (
    DocumentChunk,
    ProcessingResult,
    fixed_size_chunk,
    semantic_chunk,
    sliding_window_chunk,
    MapReduceProcessor,
    HierarchicalSummarizer,
    position_aware_assemble,
    DocumentAnalysisProcessor,
)

__all__ = [
    "DocumentChunk",
    "ProcessingResult",
    "fixed_size_chunk",
    "semantic_chunk",
    "sliding_window_chunk",
    "MapReduceProcessor",
    "HierarchicalSummarizer",
    "position_aware_assemble",
    "DocumentAnalysisProcessor",
]