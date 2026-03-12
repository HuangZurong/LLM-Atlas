import asyncio
from typing import Dict, List
from openai import AsyncOpenAI

"""
Industrial Pattern: Self-Correction (Reflexion) Chain
----------------------------------------------------
For complex tasks (code generation, legal analysis), a single prompt
often contains bugs. This pattern implements a multi-stage loop:
1. Draft: Initial response.
2. Critique: Model identifies its own mistakes.
3. Revise: Model fixes based on critique.
"""

class SelfCorrectionChain:
    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4o"):
        self.client = client
        self.model = model

    async def _call(self, system: str, user: str):
        res = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
        )
        return res.choices[0].message.content

    async def run(self, task: str):
        print(f"--- Task: {task} ---")

        # Stage 1: Initial Draft
        draft = await self._call(
            "You are an expert programmer. Write a solution to the user's task.",
            task
        )
        print("\n[STAGED 1: DRAFT COMPLETED]")

        # Stage 2: Self-Critique
        critique = await self._call(
            "You are a critical code reviewer. Find bugs, inefficiencies, or edge-case failures in the code provided.",
            f"TASK: {task}\nCODE: {draft}"
        )
        print("\n[STAGE 2: CRITIQUE GENERATED]")
        # print(f"Critique: {critique[:100]}...")

        # Stage 3: Revision
        final = await self._call(
            "You are an expert programmer. Revise your initial code based on the critique provided. Ensure it is robust.",
            f"INITIAL CODE: {draft}\nCRITIQUE: {critique}"
        )
        print("\n[STAGE 3: FINAL REVISION COMPLETED]")
        return final

async def main():
    client = AsyncOpenAI(api_key="sk-...")
    chain = SelfCorrectionChain(client)

    # Example Task: Write a regex to parse URLs
    # result = await chain.run("Write a Python function to extract domain names from URLs.")
    print("Self-Correction Chain ready.")

if __name__ == "__main__":
    asyncio.run(main())
