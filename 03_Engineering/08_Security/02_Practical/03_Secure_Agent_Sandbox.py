"""
03_Secure_Agent_Sandbox.py — Tool Execution Sandboxing & Permission Control

Implements a secure environment for agent tool execution:
1. Docker-based sandboxing simulation (gVisor/Firecracker concept)
2. Role-Based Access Control (RBAC) for tools
3. Human-in-the-loop (HITL) approval gates for destructive actions
4. Resource constraints (timeouts, memory limits)
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Permission Tiers
# ---------------------------------------------------------------------------

class ToolTier(int, Enum):
    TIER_1_READ_ONLY = 1    # Safe, auto-approve
    TIER_2_LOW_RISK_WRITE = 2 # Write to safe areas, auto-approve + log
    TIER_3_HIGH_RISK = 3     # Destructive actions, REQUIRE approval
    TIER_4_FORBIDDEN = 4     # Never allowed

@dataclass
class ToolDefinition:
    name: str
    func: Callable
    tier: ToolTier
    description: str

# ---------------------------------------------------------------------------
# Sandbox Environment
# ---------------------------------------------------------------------------

class SecureSandbox:
    """
    Simulates a secure sandbox for agent tool execution.
    """
    def __init__(self):
        self.registry: Dict[str, ToolDefinition] = {}
        self.logger = logging.getLogger(__name__)

    def register_tool(self, name: str, func: Callable, tier: ToolTier, description: str):
        self.registry[name] = ToolDefinition(name, func, tier, description)

    async def execute(self, name: str, args: Dict[str, Any], user_context: Dict[str, Any]) -> Any:
        """Execute a tool with safety checks."""
        if name not in self.registry:
            return {"error": f"Tool '{name}' not found."}

        tool = self.registry[name]

        # 1. Tier 4 Check
        if tool.tier == ToolTier.TIER_4_FORBIDDEN:
            self.logger.error(f"SECURITY ALERT: Blocked forbidden tool call: {name}")
            return {"error": "Unauthorized tool access."}

        # 2. Tier 3 Check (Human-in-the-loop)
        if tool.tier == ToolTier.TIER_3_HIGH_RISK:
            approved = await self._request_user_approval(name, args, user_context)
            if not approved:
                return {"error": "User denied tool execution."}

        # 3. Simulated Sandbox Execution
        try:
            # Simulate resource limits (timeout)
            async with asyncio.timeout(5.0):
                self.logger.info(f"Executing {name} in sandbox...")
                result = await tool.func(**args)
                return {"status": "success", "data": result}
        except asyncio.TimeoutError:
            return {"error": f"Tool {name} timed out after 5s."}
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}

    async def _request_user_approval(self, name: str, args: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Simulate a request for human approval."""
        print(f"\n[SECURITY GATE] Approval required for: {name}")
        print(f"Arguments: {json.dumps(args, indent=2)}")
        print(f"Requested by: {context.get('user_id')}")
        # In a real app, this would send a notification to a UI/Slack
        return True # Mocking auto-approval for demo

# ---------------------------------------------------------------------------
# Mock Tools
# ---------------------------------------------------------------------------

async def read_docs(query: str): return f"Found data for {query}"
async def update_record(id: int, val: str): return f"Updated {id} to {val}"
async def delete_database(): return "DELETED EVERYTHING" # Forbidden

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    sandbox = SecureSandbox()
    sandbox.register_tool("read_docs", read_docs, ToolTier.TIER_1_READ_ONLY, "Search knowledge base")
    sandbox.register_tool("update_user", update_record, ToolTier.TIER_3_HIGH_RISK, "Update user profile")
    sandbox.register_tool("drop_db", delete_database, ToolTier.TIER_4_FORBIDDEN, "Dangerous!")

    user = {"user_id": "eng_team_01", "role": "developer"}

    print("=" * 60)
    print("Secure Agent Sandbox Demo")
    print("=" * 60)

    # Test Case 1: Safe read
    print("\nCase 1: Safe Tool")
    res1 = await sandbox.execute("read_docs", {"query": "security policy"}, user)
    print(res1)

    # Test Case 2: High risk (HITL)
    print("\nCase 2: High Risk Tool (Requires Approval)")
    res2 = await sandbox.execute("update_user", {"id": 123, "val": "premium"}, user)
    print(res2)

    # Test Case 3: Forbidden
    print("\nCase 3: Forbidden Tool")
    res3 = await sandbox.execute("drop_db", {}, user)
    print(res3)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
