"""
Customer Support Scenario
=========================
Context engineering for customer support agents.

Key challenges:
- Multi-turn conversations with growing history
- Need to track order numbers, refund status, user preferences
- Token budget management as context grows
- Different compression strategies for different content types
"""

from .scenarios import (
    CustomerProfile,
    Order,
    ConversationTurn,
    SupportScenario,
    generate_support_scenario,
)
from .multi_turn_manager import SupportContextManager

__all__ = [
    "CustomerProfile",
    "Order",
    "ConversationTurn",
    "SupportScenario",
    "generate_support_scenario",
    "SupportContextManager",
]