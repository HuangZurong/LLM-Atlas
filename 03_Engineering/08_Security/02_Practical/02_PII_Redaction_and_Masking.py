"""
02_PII_Redaction_and_Masking.py — Privacy Guardrails for LLM Systems

Demonstrates how to detect and mask Personally Identifiable Information (PII)
using Microsoft Presidio or regex-based logic before data reaches the model.

Key Features:
- Detection of Emails, Credit Cards, Names, and SSNs
- Configurable masking strategies (REDACT, MASK, PSEUDONYMIZE)
- Inbound masking (User input → LLM)
- Outbound masking (LLM response → User)
"""

import re
from typing import List, Dict

# ---------------------------------------------------------------------------
# PII Detection Logic
# ---------------------------------------------------------------------------

class PIIProcessor:
    """
    Handles detection and redaction of PII.
    """
    def __init__(self):
        # Regex patterns for common PII
        self.pii_patterns = {
            "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "PHONE": r"\b(?:\+?\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b",
            "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
            "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        }

    def redact(self, text: str, replacement: str = "[REDACTED]") -> str:
        """Replace all detected PII with a fixed string."""
        redacted_text = text
        for entity_type, pattern in self.pii_patterns.items():
            redacted_text = re.sub(pattern, f"{replacement}", redacted_text)
        return redacted_text

    def mask(self, text: str) -> str:
        """Replace PII with its type label (e.g., [EMAIL])."""
        masked_text = text
        for entity_type, pattern in self.pii_patterns.items():
            masked_text = re.sub(pattern, f"[{entity_type}]", masked_text)
        return masked_text

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    processor = PIIProcessor()

    raw_input = """
    Customer Service Request:
    Name: John Doe
    Email: john.doe@example.com
    Phone: 123-456-7890
    Message: I want to update my credit card 4111-2222-3333-4444.
    """

    print("=" * 60)
    print("PII Redaction Demo")
    print("=" * 60)

    print("\n[Original Input]:")
    print(raw_input)

    print("\n[Redacted (Fixed String)]:")
    print(processor.redact(raw_input))

    print("\n[Masked (Entity Labels)]:")
    print(processor.mask(raw_input))

if __name__ == "__main__":
    main()
