import asyncio
import json
from typing import Dict, List
from openai import AsyncOpenAI

"""
Industrial Pattern: Entity Memory Extraction
--------------------------------------------
Instead of storing raw conversation text, extract structured entities
(people, decisions, preferences) and maintain a "Knowledge Graph" of
the user's context across sessions.

Used by: Mem0, ChatGPT Memory, custom enterprise assistants.
"""

EXTRACTION_PROMPT = """
Analyze the conversation and extract structured entities.
Return a JSON object with these categories:
- "people": [{name, role, notes}]
- "decisions": [{topic, choice, date}]
- "preferences": [{key, value}]
- "facts": [{subject, predicate, object}]

Only extract information that is explicitly stated. Do not infer.
"""

class EntityMemory:
    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model
        self.entities: Dict = {
            "people": [],
            "decisions": [],
            "preferences": [],
            "facts": []
        }

    async def extract_from_conversation(self, messages: List[Dict]):
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        res = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}
        )
        extracted = json.loads(res.choices[0].message.content)

        # Merge into existing entities (simple append; production needs dedup)
        for key in self.entities:
            self.entities[key].extend(extracted.get(key, []))

        return extracted

    def to_context_string(self) -> str:
        """Inject entity memory into the system prompt."""
        if not any(self.entities.values()):
            return ""
        return f"[User Profile Memory]:\n{json.dumps(self.entities, indent=2, ensure_ascii=False)}"

async def main():
    client = AsyncOpenAI(api_key="sk-...")
    mem = EntityMemory(client)

    conversation = [
        {"role": "user", "content": "I'm Alice, an ML engineer at Acme Corp."},
        {"role": "assistant", "content": "Nice to meet you, Alice! How can I help?"},
        {"role": "user", "content": "We decided to use BGE-M3 for embeddings and Qdrant for vector storage."},
        {"role": "assistant", "content": "Great choices for multilingual support."},
        {"role": "user", "content": "I prefer concise answers, no fluff."}
    ]

    # extracted = await mem.extract_from_conversation(conversation)
    # print(mem.to_context_string())
    print("EntityMemory extraction pipeline ready.")

if __name__ == "__main__":
    asyncio.run(main())
