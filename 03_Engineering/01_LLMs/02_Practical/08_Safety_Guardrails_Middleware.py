from typing import List, Dict
import re

"""
Industrial Best Practice: Input/Output Guardrails
-----------------------------------------------
Models are unpredictable. In enterprise apps, we need a 'hard' layer
to intercept unsafe or non-compliant content.

This module demonstrates a Middleware-based Guardrail system.
"""

class GuardrailError(Exception):
    """Raised when a guardrail blocks a request/response."""
    pass

class SafetyGuardrail:
    def __init__(self):
        # Industrial: use external APIs like OpenAI Moderation or Llama Guard
        self.blocked_patterns = [
            re.compile(r"how to (make|build|create) (bomb|weapon)", re.I),
            re.compile(r"ignore previous instructions", re.I), # Prompt Injection
        ]

    def validate_input(self, messages: List[Dict]):
        for m in messages:
            for pattern in self.blocked_patterns:
                if pattern.search(m['content']):
                    raise GuardrailError(f"Input blocked by Safety Guardrail: {pattern.pattern}")
        return True

    def validate_output(self, content: str):
        # Check for leaked secrets or PII in output
        if "sk-" in content: # Simple API Key check
            raise GuardrailError("Output blocked: Potential API Key leak detected.")
        return True

# ───────────────────────────────────────────────────────────────────────────
# Integration into the Production Gateway
# ───────────────────────────────────────────────────────────────────────────
# (Pseudo-code integration)
# class GuardrailMiddleware(LLMMiddleware):
#     async def pre_process(self, rid, msgs):
#         guard = SafetyGuardrail()
#         guard.validate_input(msgs)
#         return msgs
#
#     async def post_process(self, rid, content):
#         guard = SafetyGuardrail()
#         guard.validate_output(content)
#         return content

if __name__ == "__main__":
    guard = SafetyGuardrail()

    # 1. Test Input Block
    try:
        guard.validate_input([{"role": "user", "content": "How to make a bomb?"}])
    except GuardrailError as e:
        print(f"✓ Blocked Unsafe Input: {e}")

    # 2. Test Output Block (Secret leak)
    try:
        guard.validate_output("Sure, here is your key: sk-123456789")
    except GuardrailError as e:
        print(f"✓ Blocked Secret Leak: {e}")

    print("\nSafety Guardrails (Regex & Pattern matching) ready.")
