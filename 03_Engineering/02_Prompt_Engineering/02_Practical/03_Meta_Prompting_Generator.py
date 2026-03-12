import asyncio
from typing import Dict, List
from openai import AsyncOpenAI

"""
Industrial Best Practice: Meta-Prompting (System Prompt Generator)
----------------------------------------------------------------
High-quality prompts are hard to write. We can use a "Meta-Model"
(the most capable model, e.g., GPT-4o) to generate optimized system
prompts for smaller, cheaper models (e.g., GPT-4o-mini, Llama-3).

This script implements a "Prompt Architect" that takes a simple goal
and produces a structured, industrial-grade System Prompt.
"""

META_PROMPT_TEMPLATE = """
You are an expert Prompt Engineer. Your task is to write a high-performance System Prompt for an LLM based on the user's GOAL.

The System Prompt you write MUST include:
1. ROLE: A professional persona.
2. CONTEXT: Background information.
3. CONSTRAINTS: What the model MUST NOT do.
4. OUTPUT FORMAT: Precise structure (JSON/Markdown).
5. FEW-SHOT EXAMPLES: (Optional but recommended).

GOAL: {goal}

Return ONLY the generated System Prompt.
"""

class PromptArchitect:
    def __init__(self, client: AsyncOpenAI):
        self.client = client

    async def generate_system_prompt(self, goal: str) -> str:
        res = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a senior AI architect."},
                {"role": "user", "content": META_PROMPT_TEMPLATE.format(goal=goal)}
            ]
        )
        return res.choices[0].message.content

async def main():
    client = AsyncOpenAI(api_key="sk-...")
    architect = PromptArchitect(client)

    goal = "Create a customer support agent for a luxury watch brand that handles returns and repairs."
    # system_prompt = await architect.generate_system_prompt(goal)

    print("Prompt Architect (Meta-Prompting) ready.")
    print("Usage: architect.generate_system_prompt('your_goal_here')")

if __name__ == "__main__":
    asyncio.run(main())
