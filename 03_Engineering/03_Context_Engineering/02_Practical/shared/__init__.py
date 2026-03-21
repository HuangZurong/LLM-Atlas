"""
Shared Context Engineering Components
======================================
Core building blocks for context management, budgeting, and compression.
These components are scenario-agnostic and can be reused across different use cases.
"""

from .composer import ContextLayer, ContextComposer, AssembledContext, count_tokens
from .budget_controller import TokenBudgetController, BudgetConfig
from .compressor import (
    CompressionStrategy,
    TruncationStrategy,
    SlidingWindowStrategy,
    ExtractiveSummaryStrategy,
    AdaptiveCompressor,
)

__all__ = [
    "ContextLayer",
    "ContextComposer",
    "AssembledContext",
    "count_tokens",
    "TokenBudgetController",
    "BudgetConfig",
    "CompressionStrategy",
    "TruncationStrategy",
    "SlidingWindowStrategy",
    "ExtractiveSummaryStrategy",
    "AdaptiveCompressor",
]