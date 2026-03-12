"""
01_ReAct_Tool_Agent.py — Production ReAct Agent with Function Calling

Demonstrates the ReAct (Reasoning + Acting) loop using OpenAI's native
function calling API, with structured tool dispatch, max-iteration guards,
and observation injection.

Key patterns:
- Native function calling (not string-parsed "Action: ..." hacks)
- Tool registry with schema validation
- Iteration budget to prevent runaway loops
- Structured observation feedback
"""

import json
import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    """A callable tool with an OpenAI function-calling schema."""
    name: str
    description: str
    parameters: dict          # JSON Schema for the tool's input
    fn: Callable[..., Any]    # The actual implementation


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def to_openai_schema(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]


# ---------------------------------------------------------------------------
# ReAct Agent
# ---------------------------------------------------------------------------

@dataclass
class ReActAgent:
    """
    A ReAct agent that uses OpenAI function calling for tool dispatch.

    The loop:
      1. Send messages to LLM (with tools declared).
      2. If LLM returns tool_calls → execute each, append observations.
      3. If LLM returns content (no tool_calls) → done.
      4. Repeat until done or max_iterations reached.
    """
    client: AsyncOpenAI
    model: str
    system_prompt: str
    registry: ToolRegistry
    max_iterations: int = 10

    async def run(self, user_query: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query},
        ]
        tools_schema = self.registry.to_openai_schema()

        for i in range(self.max_iterations):
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools_schema if tools_schema else None,
                temperature=0,
            )
            msg = response.choices[0].message

            # --- Terminal: LLM produced a final text answer ---
            if not msg.tool_calls:
                return msg.content or ""

            # --- Non-terminal: execute tool calls ---
            messages.append(msg.model_dump())

            for call in msg.tool_calls:
                result = await self._execute_tool(call)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

        return "[Agent reached max iterations without a final answer]"

    async def _execute_tool(self, call) -> Any:
        tool = self.registry.get(call.function.name)
        if not tool:
            return {"error": f"Unknown tool: {call.function.name}"}
        try:
            args = json.loads(call.function.arguments)
            result = tool.fn(**args)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Example: Build a research agent with search + calculator
# ---------------------------------------------------------------------------

def web_search(query: str) -> dict:
    """Simulated web search — replace with real API (Tavily, SerpAPI, etc.)."""
    mock_db = {
        "python creator": {"answer": "Guido van Rossum created Python in 1991."},
        "rust creator": {"answer": "Graydon Hoare created Rust at Mozilla in 2010."},
    }
    for key, val in mock_db.items():
        if key in query.lower():
            return val
    return {"answer": f"No results found for: {query}"}


def calculator(expression: str) -> dict:
    """Evaluate a math expression safely."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return {"error": "Invalid characters in expression"}
    return {"result": eval(expression)}  # noqa: S307 — safe due to allowlist


def build_research_agent(client: AsyncOpenAI, model: str = "gpt-4o") -> ReActAgent:
    registry = ToolRegistry()

    registry.register(Tool(
        name="web_search",
        description="Search the web for factual information. Use for any question about people, events, or facts.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query"}},
            "required": ["query"],
        },
        fn=web_search,
    ))

    registry.register(Tool(
        name="calculator",
        description="Evaluate a mathematical expression. Use for any arithmetic computation.",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "Math expression, e.g. '2 + 3 * 4'"}},
            "required": ["expression"],
        },
        fn=calculator,
    ))

    return ReActAgent(
        client=client,
        model=model,
        system_prompt=(
            "You are a research assistant. Use the provided tools to answer questions accurately. "
            "Always verify facts with web_search before answering. "
            "If a calculation is needed, use the calculator tool. "
            "Cite your sources in the final answer."
        ),
        registry=registry,
        max_iterations=6,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    client = AsyncOpenAI()  # Uses OPENAI_API_KEY env var
    agent = build_research_agent(client)

    queries = [
        "Who created Python and in what year?",
        "What is (1991 - 1956) * 3?",
        "Who created Rust and how many years after Python was it created?",
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        answer = await agent.run(q)
        print(f"A: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
