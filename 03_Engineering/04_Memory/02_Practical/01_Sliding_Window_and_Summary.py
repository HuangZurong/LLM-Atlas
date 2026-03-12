import asyncio
import tiktoken
from typing import List, Dict
from openai import AsyncOpenAI

"""
Industrial Pattern: Hybrid Memory Manager (Sliding Window + Summary)
-------------------------------------------------------------------
The most common production pattern for multi-turn chat:
1. Keep the last N turns verbatim (recency).
2. Summarize older turns into a compressed paragraph (history).
3. Always keep the System Prompt intact.
"""

class HybridMemoryManager:
    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4o-mini",
                 max_context_tokens: int = 16000, recent_turns: int = 6):
        self.client = client
        self.model = model
        self.enc = tiktoken.encoding_for_model("gpt-4o")
        self.max_context_tokens = max_context_tokens
        self.recent_turns = recent_turns  # Keep last N messages verbatim
        self.summary = ""
        self.full_history: List[Dict] = []

    def _count_tokens(self, messages: List[Dict]) -> int:
        return sum(len(self.enc.encode(m['content'])) for m in messages)

    async def _summarize(self, messages: List[Dict]) -> str:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        res = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Summarize this conversation history in 2-3 sentences. Preserve key facts, decisions, and user preferences."},
                {"role": "user", "content": text}
            ],
            max_tokens=300
        )
        return res.choices[0].message.content

    async def add_turn(self, role: str, content: str):
        self.full_history.append({"role": role, "content": content})

    async def build_context(self, system_prompt: str) -> List[Dict]:
        """Assemble the context window with budget management."""
        messages = [{"role": "system", "content": system_prompt}]

        # Inject summary of older turns
        if self.summary:
            messages.append({"role": "system", "content": f"[Conversation Summary]: {self.summary}"})

        # Add recent turns verbatim
        recent = self.full_history[-self.recent_turns:]
        messages.extend(recent)

        # Check if we're over budget
        if self._count_tokens(messages) > self.max_context_tokens:
            # Compress: summarize everything except the last few turns
            old_turns = self.full_history[:-self.recent_turns]
            if old_turns:
                self.summary = await self._summarize(old_turns)
            # Rebuild
            messages = [{"role": "system", "content": system_prompt}]
            if self.summary:
                messages.append({"role": "system", "content": f"[Conversation Summary]: {self.summary}"})
            messages.extend(recent)

        return messages

async def main():
    client = AsyncOpenAI(api_key="sk-...")
    mem = HybridMemoryManager(client, max_context_tokens=8000, recent_turns=4)

    # Simulate a conversation
    await mem.add_turn("user", "I'm building a RAG system for legal documents.")
    await mem.add_turn("assistant", "Great. What embedding model are you considering?")
    await mem.add_turn("user", "BGE-M3, because we need multilingual support.")
    await mem.add_turn("assistant", "Good choice. Let's discuss chunking strategy next.")

    # context = await mem.build_context("You are a helpful AI assistant.")
    # print(context)
    print("HybridMemoryManager (Sliding Window + Summary) ready.")

if __name__ == "__main__":
    asyncio.run(main())
