"""
Customer Support Scenario Data
==============================
Realistic customer support data for demonstrating context engineering.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CustomerProfile:
    """Customer information."""
    name: str
    customer_id: str
    membership_tier: str  # "standard", "premium", "business"
    member_since: str
    preferred_contact: str = "email"
    past_issues: list[str] = field(default_factory=list)
    sentiment_history: str = "generally satisfied"


@dataclass
class Order:
    """Order information."""
    order_id: str
    product: str
    price: float
    order_date: str
    status: str  # "processing", "shipped", "delivered", "refunded"
    tracking_number: Optional[str] = None
    estimated_delivery: Optional[str] = None


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str  # "user" or "assistant"
    content: str


@dataclass
class SupportScenario:
    """Complete support scenario with all context."""
    customer: CustomerProfile
    orders: list[Order]
    conversation_history: list[ConversationTurn]
    current_query: str
    knowledge_base: str  # RAG context


def generate_support_scenario() -> SupportScenario:
    """
    Generate a realistic customer support scenario.

    Scenario: Jane Doe, a Premium member, has issues with order #AC-7823
    (delayed shipping, refund request) and wants to check order #AC-7901.
    She also wants to return an older order #AC-7654 (past 30-day window).
    """

    customer = CustomerProfile(
        name="Jane Doe",
        customer_id="CUST-4521",
        membership_tier="premium",
        member_since="2022-03-15",
        preferred_contact="email",
        past_issues=[
            "2 refund requests (both approved)",
            "One escalation in 2023 (resolved)",
        ],
        sentiment_history="generally satisfied, occasional frustration with shipping delays",
    )

    orders = [
        Order(
            order_id="AC-7823",
            product="Wireless Keyboard (Model XK-500)",
            price=89.00,
            order_date="2025-03-10",
            status="shipped",
            tracking_number="1Z999AA10123456784",
            estimated_delivery="2025-03-15",
        ),
        Order(
            order_id="AC-7901",
            product="Laptop Stand (Ergonomic Pro)",
            price=45.00,
            order_date="2025-03-14",
            status="processing",
        ),
        Order(
            order_id="AC-7654",
            product="Gaming Headset (Model GH-200)",
            price=129.00,
            order_date="2025-02-08",
            status="delivered",
        ),
    ]

    conversation_history = [
        ConversationTurn("user", "Hi, I placed an order last week and haven't received a shipping confirmation."),
        ConversationTurn("assistant", "Hello Jane! I can look into that for you. Could you provide your order number?"),
        ConversationTurn("user", "It's order #AC-7823."),
        ConversationTurn("assistant", "Thank you. Order #AC-7823 was placed on March 10th for a Wireless Keyboard ($89). It shipped on March 12th via FedEx. Tracking: 1Z999AA10123456784."),
        ConversationTurn("user", "The tracking hasn't updated in 5 days. It's stuck in Memphis."),
        ConversationTurn("assistant", "I can see the package has been stationary since March 13th. This may indicate a carrier delay. I've flagged it with FedEx for investigation."),
        ConversationTurn("user", "That's frustrating. What if it's lost?"),
        ConversationTurn("assistant", "If FedEx confirms the package is lost, we will immediately ship a replacement at no cost or issue a full refund."),
        ConversationTurn("user", "I'd prefer a refund at this point."),
        ConversationTurn("assistant", "Understood. I've noted your preference for a refund. Once FedEx responds, I'll process it immediately."),
        ConversationTurn("user", "It's been 48 hours and no update from FedEx."),
        ConversationTurn("assistant", "I apologize for the delay. I've escalated this directly to our logistics team. As a Premium member, I'm authorizing an express refund of $89 now."),
        ConversationTurn("user", "Thank you. Also, I ordered something else — order #AC-7901. Can you check that too?"),
        ConversationTurn("assistant", "Order #AC-7901 is a Laptop Stand ($45), placed March 14th. It's currently processing and will ship within 1-2 business days."),
        ConversationTurn("user", "Great. One more thing — I want to return a headset I bought in February. Order #AC-7654."),
        ConversationTurn("assistant", "Order #AC-7654 was placed on February 8th. Our 30-day return window expired on March 10th. I'm unable to process a standard return."),
        ConversationTurn("user", "That's disappointing. I never even opened the box."),
        ConversationTurn("assistant", "I understand your frustration. As a valued Premium member, I can offer a one-time exception and process the return."),
        ConversationTurn("user", "Perfect, thank you so much."),
        ConversationTurn("assistant", "You're welcome, Jane! Is there anything else I can help you with today?"),
    ]

    current_query = "I haven't received the refund for order #AC-7823 yet. It's been 3 days. Can you check the status?"

    knowledge_base = """
[Refund Policy - FAQ]

Standard refunds are processed within 5-7 business days after the item is received at our warehouse.
Items must be returned within 30 days of the original purchase date.
Digital products and downloadable software are non-refundable once accessed.
Damaged items require photo evidence submitted via the support portal within 48 hours of delivery.

Express refunds (24h processing) are available for Premium members at no extra charge.
For orders over $500, a manager approval is required before processing any refund.
Shipping costs are non-refundable unless the return is due to our error (wrong item, damaged, defective).

Refund methods:
- Original payment method: 5-7 business days
- Store credit: Immediate
- Express refund (Premium only): 24 hours

If a package is lost in transit, we will either:
1. Ship a replacement at no cost, OR
2. Issue a full refund immediately

For Premium members: We offer one-time exceptions for returns outside the 30-day window.
"""

    return SupportScenario(
        customer=customer,
        orders=orders,
        conversation_history=conversation_history,
        current_query=current_query,
        knowledge_base=knowledge_base,
    )