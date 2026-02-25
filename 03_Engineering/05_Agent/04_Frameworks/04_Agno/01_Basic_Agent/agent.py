"""
Agno Basic Agent Example

This example demonstrates the core Agno patterns:
1. Simple agent creation
2. Tool integration with automatic schema generation
3. Async streaming responses
4. Built-in observability

Prerequisites:
    pip install agno openai
"""

import asyncio
from typing import Optional

from agno import Agent, Tool

# ---------------------------------------------------------------------------
# Define Tools
# ---------------------------------------------------------------------------

@Tool
def get_current_time(timezone: str = "UTC") -> dict:
    """
    Get the current time in a specific timezone.

    Args:
        timezone: IANA timezone name (e.g., "America/New_York")

    Returns:
        Dictionary with timezone and current time
    """
    from datetime import datetime
    import pytz

    try:
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        return {
            "status": "success",
            "timezone": timezone,
            "current_time": current_time,
            "message": f"Current time in {timezone} is {current_time}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Invalid timezone: {timezone}. Error: {str(e)}"
        }


@Tool
def calculate(expression: str) -> dict:
    """
    Evaluate a mathematical expression.

    Args:
        expression: Mathematical expression (e.g., "2 + 3 * 4")

    Returns:
        Dictionary with result and calculation details
    """
    import ast
    import operator
    import math

    # Safe evaluation with limited operations
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    allowed_names = {
        'pi': math.pi,
        'e': math.e,
    }

    class SafeEval(ast.NodeVisitor):
        def visit_BinOp(self, node):
            left = self.visit(node.left)
            right = self.visit(node.right)
            return allowed_operators[type(node.op)](left, right)

        def visit_UnaryOp(self, node):
            operand = self.visit(node.operand)
            return allowed_operators[type(node.op)](operand)

        def visit_Num(self, node):
            return node.n

        def visit_Name(self, node):
            if node.id in allowed_names:
                return allowed_names[node.id]
            raise ValueError(f"Disallowed name: {node.id}")

        def visit_Expr(self, node):
            return self.visit(node.value)

        def generic_visit(self, node):
            raise ValueError(f"Disallowed operation: {type(node).__name__}")

    try:
        tree = ast.parse(expression, mode='eval')
        evaluator = SafeEval()
        result = evaluator.visit(tree.body)

        return {
            "status": "success",
            "expression": expression,
            "result": result,
            "type": type(result).__name__
        }
    except Exception as e:
        return {
            "status": "error",
            "expression": expression,
            "error": str(e),
            "message": "Calculation failed. Use basic arithmetic operators only."
        }


@Tool
def search_knowledge_base(query: str, limit: int = 3) -> dict:
    """
    Search a mock knowledge base for information.

    Args:
        query: Search query
        limit: Maximum number of results to return

    Returns:
        Dictionary with search results
    """
    # Mock knowledge base - in production, connect to real data
    knowledge_base = {
        "python": [
            "Python is a high-level, interpreted programming language.",
            "Python supports multiple programming paradigms.",
            "Python has a large standard library and active community.",
        ],
        "machine learning": [
            "Machine learning is a subset of artificial intelligence.",
            "Common ML algorithms include linear regression, decision trees, and neural networks.",
            "Deep learning uses multi-layer neural networks.",
        ],
        "web development": [
            "Web development involves building websites and web applications.",
            "Common technologies include HTML, CSS, JavaScript, and various frameworks.",
            "Backend development typically uses languages like Python, Java, or Node.js.",
        ]
    }

    query_lower = query.lower()
    results = []

    for category, facts in knowledge_base.items():
        if category in query_lower:
            results.extend(facts[:limit])

    if not results:
        results = ["No specific information found. Try more specific queries."]

    return {
        "status": "success",
        "query": query,
        "results": results[:limit],
        "count": len(results[:limit])
    }


# ---------------------------------------------------------------------------
# Create Agents
# ---------------------------------------------------------------------------

def create_basic_agent() -> Agent:
    """Create a basic assistant agent with time and calculation tools."""
    return Agent(
        name="BasicAssistant",
        instructions="""
        You are a helpful assistant that can:
        1. Tell the current time in any timezone
        2. Perform mathematical calculations
        3. Search a knowledge base for information

        Be concise and accurate. Always use the appropriate tool when needed.
        If a user asks for something outside your capabilities, say so politely.
        """,
        model="gpt-4o-mini",  # Use a cheaper model for basic tasks
        tools=[get_current_time, calculate, search_knowledge_base],
        show_tool_calls=True,  # Debug: show when tools are called
    )


def create_advanced_agent() -> Agent:
    """Create an advanced agent for complex reasoning."""
    return Agent(
        name="AdvancedAssistant",
        instructions="""
        You are an advanced AI assistant capable of complex reasoning and analysis.

        Guidelines:
        - Break down complex problems into steps
        - Use tools when appropriate, but also provide reasoning
        - Consider edge cases and provide thorough explanations
        - When searching, synthesize information from multiple sources

        You have access to time, calculation, and knowledge base tools.
        """,
        model="gpt-4o",  # Use a more capable model
        tools=[get_current_time, calculate, search_knowledge_base],
        temperature=0.3,  # Lower temperature for more consistent results
    )


# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------

async def run_examples():
    """Run example queries through both agents."""

    print("=" * 60)
    print("Agno Agent Examples")
    print("=" * 60)

    # Create agents
    basic_agent = create_basic_agent()
    advanced_agent = create_advanced_agent()

    test_cases = [
        {
            "query": "What time is it in Tokyo?",
            "agent": "basic",
            "description": "Simple tool use - time lookup"
        },
        {
            "query": "Calculate (15 * 3) + (8 / 2) - 7",
            "agent": "basic",
            "description": "Mathematical calculation"
        },
        {
            "query": "What is machine learning?",
            "agent": "basic",
            "description": "Knowledge base search"
        },
        {
            "query": "Explain the difference between AI and machine learning, and calculate the square root of 144",
            "agent": "advanced",
            "description": "Complex multi-step query"
        },
        {
            "query": "What's the time in London, and tell me about Python programming",
            "agent": "advanced",
            "description": "Multiple tool calls in one query"
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {test['description']}")
        print(f"Query: {test['query']}")
        print(f"Agent: {test['agent'].title()}")
        print(f"{'-'*60}")

        # Select agent
        agent = basic_agent if test['agent'] == 'basic' else advanced_agent

        try:
            # Run the query
            response = await agent.run(test['query'])
            print(f"Response:\n{response}")
        except Exception as e:
            print(f"Error: {e}")

    # Demonstrate streaming
    print(f"\n{'='*60}")
    print("Streaming Example")
    print(f"{'-'*60}")
    print("Query: Explain machine learning step by step")

    print("\nStreaming response:")
    async for chunk in basic_agent.stream("Explain machine learning step by step"):
        print(chunk, end="", flush=True)
    print()  # Newline after streaming


async def compare_agents():
    """Compare basic vs advanced agent responses."""
    query = "What is the capital of France and what time is it there?"

    print(f"\n{'='*60}")
    print("Agent Comparison")
    print(f"{'-'*60}")
    print(f"Query: {query}")

    basic_agent = create_basic_agent()
    advanced_agent = create_advanced_agent()

    print("\n1. Basic Agent Response:")
    basic_response = await basic_agent.run(query)
    print(basic_response)

    print("\n2. Advanced Agent Response:")
    advanced_response = await advanced_agent.run(query)
    print(advanced_response)


# ---------------------------------------------------------------------------
# Production Patterns
# ---------------------------------------------------------------------------

async def production_patterns():
    """Demonstrate production-ready patterns."""

    print(f"\n{'='*60}")
    print("Production Patterns")
    print(f"{'-'*60}")

    # 1. Error handling
    agent = create_basic_agent()

    try:
        response = await agent.run("Calculate 10 / 0")
        print(f"Response to invalid calculation: {response}")
    except Exception as e:
        print(f"Error handled gracefully: {e}")

    # 2. Timeout handling
    import asyncio
    from asyncio import TimeoutError

    try:
        async with asyncio.timeout(5.0):  # 5 second timeout
            response = await agent.run("What is machine learning?")
            print(f"\nResponse with timeout: {response[:100]}...")
    except TimeoutError:
        print("\nRequest timed out after 5 seconds")

    # 3. Batch processing
    print("\nBatch processing example:")
    queries = [
        "Time in New York",
        "Calculate 2 + 2",
        "What is Python?",
    ]

    tasks = [agent.run(q) for q in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, (query, result) in enumerate(zip(queries, results)):
        if isinstance(result, Exception):
            print(f"Query {i+1} failed: {query} → Error: {result}")
        else:
            print(f"Query {i+1}: {query} → {result[:50]}...")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    """Run all examples."""
    print("Starting Agno Agent Examples...")

    # Set your OpenAI API key (in production, use environment variables)
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Some examples may fail.")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")

    await run_examples()
    await compare_agents()
    await production_patterns()

    print(f"\n{'='*60}")
    print("All examples completed!")
    print("Next steps:")
    print("1. Install Agno: pip install agno")
    print("2. Set API key: export OPENAI_API_KEY='your-key'")
    print("3. Run: python agent.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())